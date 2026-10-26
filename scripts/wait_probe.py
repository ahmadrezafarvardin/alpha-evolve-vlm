"""Does the Thinking model's answer change after it second-guesses itself ("Wait, ...")? Intervention measurement.

    python scripts/wait_probe.py scores/dev/scores_manual_thinking_s1.json analysis/wait_probe_dev.json --k 1 2 4

For each recorded response of manual_thinking, cut the reasoning just before its k-th "Wait", force the answer
exactly as manual_thinking does when its budget runs out, and generate the answer greedily. Responses with fewer
than k "Wait"s keep their recorded answer (the cut would not change them). Compares accuracy and reasoning tokens
with the recorded run. Measurement only (no program changes); the recorded reasoning is re-tokenized from text,
so the prefix can differ from the original token ids at a few positions.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import manual_thinking as m  # noqa: E402
from evaluate import load_charxiv_data  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402

WAIT = re.compile(r"\bWait\b")
FORCED_MARK = "I have to give the final answer now."


def reasoning_of(raw):
    return raw.split("</think>")[0].split("\n\n" + FORCED_MARK)[0]


def answer_after_cut(model, processor, query, reasoning_prefix):
    question = query["question"]
    prompt = question + m._ANSWER_ONLY + (m._CHECK_NOT_APPLICABLE if "Not Applicable" in question else "")
    messages = [{"role": "user", "content": [{"type": "image", "image": query["figure_path"]},
                                             {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text += reasoning_prefix.rstrip() + m._FORCE_ANSWER
    images, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(model.device)
    out = m._generate(model, inputs, inputs.input_ids, m._ANSWER_TOKENS)
    return m.postprocess(question, processor.tokenizer.decode(out[0], skip_special_tokens=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scores")
    parser.add_argument("out")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()

    scores = json.loads(Path(args.scores).read_text())
    queries, _ = load_charxiv_data(None, scores["split"])
    model, processor = m._get_model()
    tok = processor.tokenizer
    rows = []
    for r in scores["responses"]:
        reasoning = reasoning_of(r["raw_response"])
        waits = [w.start() for w in WAIT.finditer(reasoning)]
        row = {"key": r["key"], "qid": r["qid"], "gold": r["gold"], "final": r["response"], "correct": r["correct"],
               "forced": FORCED_MARK in r["raw_response"], "waits": len(waits),
               "reasoning_tokens": len(tok(reasoning).input_ids), "cut": {}}
        for k in args.k:
            if len(waits) < k:
                row["cut"][k] = {"answer": r["response"], "correct": r["correct"], "tokens": row["reasoning_tokens"]}
                continue
            prefix = reasoning[:waits[k - 1]]
            answer = answer_after_cut(model, processor, queries[r["key"]], prefix)
            row["cut"][k] = {"answer": answer, "correct": answer.strip().lower() == r["gold"].lower(),
                             "tokens": len(tok(prefix).input_ids)}
        rows.append(row)
    Path(args.out).write_text(json.dumps({"source": args.scores, "k": args.k, "rows": rows}, indent=1))

    n = len(rows)
    print(f"{args.scores}: {n} responses, recorded accuracy {sum(r['correct'] for r in rows)}/{n}, "
          f"mean reasoning tokens {sum(r['reasoning_tokens'] for r in rows) / n:.0f}")
    for k in args.k:
        cut = [r for r in rows if r["waits"] >= k]
        acc = sum(r["cut"][k]["correct"] for r in rows)
        flips_bad = sum(r["correct"] and not r["cut"][k]["correct"] for r in cut)
        flips_good = sum(not r["correct"] and r["cut"][k]["correct"] for r in cut)
        tokens = sum(r["cut"][k]["tokens"] for r in rows) / n
        print(f"  cut at Wait #{k}: {len(cut)} responses cut; accuracy {acc}/{n}; "
              f"right->wrong {flips_bad}, wrong->right {flips_good}; mean reasoning tokens {tokens:.0f}")


if __name__ == "__main__":
    main()
