"""Mutation prompt: parent program + its evo results, inspirations, and sampled improvement directions.

Only `evo` results ever appear here (CONTRACT.md §4): failure examples come from the parent's full-stage
evo scores file. Prompts are stochastic (AlphaEvolve): the set and order of suggested directions, the number of
failure examples and the requested change size are sampled per child, and recorded with the LLM call.
"""
import difflib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluate import QID_CATEGORY, load_charxiv_data  # noqa: E402

SYSTEM = """You are an expert in efficient vision-language model inference, improving a Python function that \
answers chart questions (CharXiv descriptive questions) with {model_id}.

Goal: higher exact-match accuracy first, then lower time per query. A response is correct only if it equals the \
gold answer exactly after .strip(), case-insensitive.

Hard rules (a violating program is rejected without evaluation):
- Keep `vlm_inference(image_path, question) -> str` working as the module's entry point; it is called once per query.
- Greedy decoding only: do_sample=False as a literal, no temperature/top_p/top_k sampling, no beam search.
- Only the model {model_id} (quantized variants allowed). No other models, no external APIs.
- Never read benchmark files (charxiv/data, answers) or import the evaluator.
- Change only code between # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END.

Answer format: a short explanation of the idea, then one or more edits, each exactly:
<<<<<<< SEARCH
exact, contiguous lines copied from the current program
=======
replacement lines
>>>>>>> REPLACE
Each SEARCH text must appear exactly once in the program."""

DIRECTIONS = [
    "Prompt wording: per question type (the question text identifies it), give tailored instructions for the answer "
    "format, e.g. layout as 'n by m', tick values exactly as written, legend labels comma-separated in order.",
    "'Not Applicable': the question's own condition decides it. Make the model check that condition reliably for the "
    "question types where it errs, without answering 'Not Applicable' when the element exists.",
    "Counting and value-reading questions (lines, ticks, legend entries, tick differences): allow a short, bounded "
    "reasoning step only for those question types, then extract the final answer robustly.",
    "Image resolution: set min_pixels/max_pixels on the processor or in the message (Qwen-VL vision tokens scale with "
    "resolution). Higher can help reading small tick labels; lower is faster.",
    "Generation budget: tune max_new_tokens per question type; stop early when the answer is complete.",
    "Answer post-processing: normalize the answer format (numbers, notation such as 10^-3, quotes, trailing "
    "punctuation, units the question says to drop) without changing correct answers.",
    "System prompt: add a concise system message that states the exact-answer convention.",
    "Speed: avoid redundant work per call (e.g. reuse processed inputs when the same image is asked about again, "
    "cheaper preprocessing), without changing outputs.",
]


# Thinking model: bounding the reasoning comes first in every prompt (user decision 2026-10-14, option (b)). In
# manual_thinking step 1, 36 % of responses exhaust the 1,024-token budget and take two thirds of the time, mostly
# second-guessing ("Wait, ..."). The other directions are sampled as for Instruct.
THINKING_FIRST = (
    "Bound the reasoning (largest cost): most time goes to responses that second-guess themselves ('Wait, ...') until "
    "the token budget runs out, and these are mostly wrong. Consider: stopping the reasoning early once it starts "
    "second-guessing (e.g. before the k-th 'Wait'), then forcing the closing </think> tag and a short answer; a "
    "lower or per-question-type reasoning budget with the same forced close; or taking the answer from the first "
    "confident statement in the reasoning. Stopping can be done during generation (e.g. a StoppingCriteria on the "
    "decoded text) so the tokens are never generated."
)
THINKING_DIRECTIONS = [
    "Prompt wording: per question type (the question text identifies it), give tailored instructions for the final "
    "answer format, e.g. layout as 'n by m', tick values exactly as written, legend labels comma-separated in order.",
    "'Not Applicable': the question's own condition decides it. Make the model check that condition reliably for the "
    "question types where it errs, without answering 'Not Applicable' when the element exists.",
    "Image resolution: set min_pixels/max_pixels on the processor or in the message (vision tokens scale with "
    "resolution). Fewer vision tokens make every reasoning token cheaper; higher can help reading small labels.",
    "Answer post-processing: normalize the final answer after </think> (numbers, notation such as 10^-3, quotes, "
    "trailing punctuation) without changing correct answers.",
    "Speed: avoid redundant work per call (reuse processed inputs for the same image, cheaper preprocessing, "
    "efficient generation settings), without changing outputs.",
]


class PromptBuilder:
    def __init__(self, model_id, rng):
        self.model_id, self.rng = model_id, rng
        self.questions = {k: q["question"] for k, q in load_charxiv_data(None, "evo")[0].items()}

    def system(self):
        return SYSTEM.format(model_id=self.model_id)

    def _scores_block(self, s):
        cats = ", ".join(f"{c} {v['accuracy']:.0%}" for c, v in sorted((s.get("per_category") or {}).items()))
        return f"accuracy {s['accuracy']:.1%}, {s['avg_time_per_query']:.2f} s/query; by category: {cats}"

    @staticmethod
    def _diff(parent_code, code):
        """Unified diff from the parent to an inspiration: the same ideas as the full code, at a fraction of the tokens."""
        lines = list(difflib.unified_diff(parent_code.splitlines(), code.splitlines(), "current", "other", n=2,
                                          lineterm=""))
        return "```diff\n" + "\n".join(lines[2:]) + "\n```" if lines else "(identical to the current program)"

    def _failures(self, scores_file, k):
        responses = json.loads(Path(scores_file).read_text())["responses"]
        wrong = [r for r in responses if not r["correct"]]
        lines = []
        for r in self.rng.sample(wrong, min(k, len(wrong))):
            question = self.questions[r["key"]].split("\n")[0]
            lines.append(f"- [{QID_CATEGORY[r['qid']]}] Q: {question}\n  model: {r['response'][:120]!r}  gold: {r['gold']!r}")
        return "\n".join(lines)

    @staticmethod
    def _tried_block(tried):
        """Edits already proposed from this parent and what came of them, so the LLM does not repeat them."""
        lines = [f"- {idea} -> {outcome}" for idea, outcome in tried]
        return ("## Already tried from the current program (do not repeat these; propose something different)\n"
                + "\n".join(lines)) if lines else ""

    def build(self, parent_code, parent_scores, parent_scores_file, inspirations, tried=()):
        """(prompt text, sampled settings). `inspirations`: [(code, full-stage summary), ...], shown as diffs;
        `tried`: [(one-line idea, outcome), ...] of earlier children of the same parent (no random draws)."""
        if "Thinking" in self.model_id:
            sampled = self.rng.sample(THINKING_DIRECTIONS, self.rng.randint(1, 3))
            directions = [THINKING_FIRST] + sampled
            direction_ids = ["bound_reasoning"] + [f"t{THINKING_DIRECTIONS.index(d)}" for d in sampled]
        else:  # Instruct: unchanged, so a resumed Instruct run draws the same prompts
            directions = self.rng.sample(DIRECTIONS, self.rng.randint(2, 4))
            direction_ids = [DIRECTIONS.index(d) for d in directions]
        n_failures = self.rng.choice([6, 10, 14])
        size = self.rng.choice(["a focused change to one aspect", "a focused change to one aspect",
                                "a bolder change that combines two ideas"])
        parts = [f"## Current program\nResults on the evaluation set: {self._scores_block(parent_scores)}\n"
                 f"```python\n{parent_code}\n```"]
        for i, (code, scores) in enumerate(inspirations, 1):
            parts.append(f"## Other program {i} (for ideas; do not edit it)\nResults: {self._scores_block(scores)}\n"
                         f"How it differs from the current program:\n{self._diff(parent_code, code)}")
        parts.append(f"## Some wrong answers of the current program\n{self._failures(parent_scores_file, n_failures)}")
        if tried:
            parts.append(self._tried_block(tried))
        parts.append("## Directions worth considering\n" + "\n".join(f"- {d}" for d in directions))
        parts.append(f"## Task\nPropose {size} to the current program that should raise accuracy or cut time "
                     "without hurting accuracy. Reply with the explanation and SEARCH/REPLACE edits only.")
        settings = {"directions": direction_ids, "n_failures": n_failures, "size": size}
        return "\n\n".join(parts), settings
