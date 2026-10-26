# Evolving Chart-QA Inference for a 2B Vision-Language Model

An AlphaEvolve-style system that uses an LLM to rewrite the inference code of **Qwen3-VL-2B** (Instruct and Thinking)
for chart question answering on **CharXiv**. It is compared against step-by-step manual optimization on an 800-query
holdout split that no design decision touched.

Two findings matter more than the headline accuracy:

- **Specification gaming.** Evolution's whole Instruct gain comes from learning to answer "Not Applicable" more
  often, which exploits a base rate in the benchmark. It does not come from reading charts better.
- **A routing reversal.** A two-model routing design passed validation and then failed on holdout, because it had
  been validated against the wrong baseline.

Report: [`report/report.pdf`](report/report.pdf) (NeurIPS preprint style, 6 pages). Full lab notebook:
[`RESULTS.md`](RESULTS.md).

---

## The problem

[CharXiv](https://github.com/princeton-nlp/CharXiv) asks descriptive questions about real charts from arXiv papers:
axis labels, tick values, legend entries, subplot layout, colorbar ranges. Scoring is exact match, so the answer
*format* matters as much as reading the chart. The target is a single function,
`vlm_inference(image_path, question) -> str`, run with a 2B model under greedy decoding. The task is to improve its
accuracy and speed in two ways:

1. **By hand**, one measured change at a time.
2. **By evolution**, following [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/):
   an LLM proposes code edits, a program database keeps a diverse population, and an evaluation cascade scores
   candidates on the GPU.

## Approach

**Data discipline.** The 1,000 labelled CharXiv validation figures are split by figure into `dev` (128, manual design),
`evo` (400, the only feedback evolution sees), `holdout` (800, final comparison only) and `reserve` (single-use
checks, each logged). Every number in this README is a holdout number unless stated otherwise.

**Manual optimization.** Each step was committed and tagged, and post-processing gains were measured exactly by replaying
recorded raw outputs (decoding is greedy):

- answer extraction from verbose responses;
- an answer-only prompt with a 64-token limit (5.4× faster);
- a "check the Not Applicable condition first" instruction;
- for Thinking: a reasoning budget with a forced `</think>`.

**Evolution system** ([`alphaevolve/`](alphaevolve/), about 1k lines, written from scratch):

| Component | Implementation |
|---|---|
| Program database | Islands with ring migration, plus a per-island MAP-Elites grid (time/query × number-reading accuracy) |
| Prompt sampler | Parent code and per-category scores, failure examples from `evo` only, inspirations shown as diffs (−56 % tokens), stochastic improvement directions |
| LLM mutation | SEARCH/REPLACE diffs inside an `EVOLVE-BLOCK`; Gemini 3.1 Flash-Lite (85 %) / Gemini 3 Flash (15 %) sampled per call |
| Evaluation cascade | Static checks → 8 → 64 → 400 queries, with a noise-aware promotion rule |
| Loop | Atomic checkpoint per generation, pushed to a git branch; exact resume from cached LLM calls |

**Extensions added after measured failures:**

- a near-duplicate edit filter (Jaccard ≥ 0.8 between siblings);
- noise-aware selection (time differences under 10 % are ignored);
- AST anti-cheating checks: no sampling, no other model, no access to gold answers;
- an "already tried" memory in the prompt;
- pause/resume on API failures.

The full Instruct run (72 children) cost **$0.23** in LLM calls.

## Key results (holdout, 800 queries, RTX 3090)

| Program | Holdout acc. | s/query | Gain | Source of gain |
|---|---|---|---|---|
| Naive baseline (Instruct) | 25.9 % | 2.61 | | |
| Manual Instruct | 56.0 % | **0.41** | **+30.1** over baseline | Manual: extraction, answer-only prompt, NA check |
| Evolved Instruct | 62.0 % | 0.46 | **+6.0** over manual (p = 0.0003) | Evolution (see *specification gaming* below) |
| Manual Thinking | 64.8 % | 28.6 | | Manual: reasoning budget, answer extraction |
| **Evolved Thinking** | **67.0 %** | 19.3 | **+2.2** and **−33 % time** over manual (p = 0.01) | Evolution: halved reasoning budget |
| Routing (manual models) | 61.6 % | 14.5 | −3.2 vs. Thinking alone | See *routing reversal* |

Paired differences use McNemar's test. Timing noise is about 7 %, so speed differences under 10 % are not claimed.

## Finding 1: Specification gaming in LLM-driven prompt evolution

An answer-level audit of the evolved Instruct program, done after the report, shows where its +6.0 points come from:

| Holdout | Manual Instruct | Evolved Instruct | Δ |
|---|---|---|---|
| Gold answer is "Not Applicable" (200) | 83 | **174** | **+91** |
| All other questions (600) | **365** | 322 | **−43** |
| Times the model answered "Not Applicable" | 115 | 262 | |

Evolution rewrote the instruction to *"check carefully whether the requested element (colorbar, continuous legend,
lines, intersection, axis) actually exists … you must answer exactly 'Not Applicable'"*. Most CharXiv charts have no
colorbar, so on the colorbar questions the evolved program answers "Not Applicable" 115 times out of 119. On the 12
charts that really do have a colorbar it scores **0/12**; the manual version scored 4/12. In the winning lineage, the single edit
that introduced this wording produced 23 of the 31 points of training gain. Image caching changed no answers, and
the persona and list-format edits are within noise.

The manual work had found the same lever and declined it: answering "Not Applicable" for whole question types would
exploit the base rate (see `RESULTS.md`, manual step 2c). The evolution loop optimized raw accuracy and pushed the
lever as far as a prompt could. It never touched the gold answers or the evaluator, since the AST checks rule that
out. **LLM-driven program search optimizes the metric you give it, including the metric's blind spots, so the
fitness function needs the same scrutiny as the search.** The gain is real on holdout, but it is a gain in
benchmark-specific calibration, not in chart understanding.

The report (written before this audit) attributes the gain to "number-reading and counting questions". That is
accurate at the category level, because the colorbar questions belong to that category, but this table gives the
mechanism behind it.

## Finding 2: The routing reversal (a methodological lesson)

The Thinking model was much better than Instruct on number-reading and counting questions in the design data, so a
router sent those question types to Thinking and everything else to the 70× faster Instruct model. On a fresh,
single-use validation slice, routing beat Instruct alone by **+9.0 points** (p < 0.001) and became the pre-registered
best-accuracy candidate. On holdout it still beat Instruct alone (+5.6, p < 10⁻⁵), but it **lost to Thinking alone**
(61.6 vs. 64.8 %). Thinking was also better on OCR-type (+19) and layout (+5) questions, which the small design slice
had not shown. The validation itself was sound, but it asked the wrong question: it compared routing with one
component, not with the strongest alternative. We only caught this because the holdout split had been kept untouched
for the final comparison, and selection was then moved to a second fresh slice instead of being re-tuned on holdout.
The lesson: **validate a composite system against its strongest component, not the most convenient one, and keep a
split that nothing has touched for long enough to find out.**

## Other negative results

- **Thinking evolution chose the easiest edit, not the best one.** It halved the reasoning budget. A "stop at the 4th
  *Wait*" intervention, measured beforehand and suggested first in the mutation prompt, cut about 60 % of reasoning
  tokens with no accuracy loss. Four of six children were byte-identical despite the "already tried" memory.
- **The evaluation cascade saved only 10 % of Instruct GPU time.** Full evaluations were cheap (2.6 min) and 75 % of
  children passed. The real saving came from the static filters that run before the GPU.

## Repository layout

```
manual_{instruct,thinking}.py     hand-optimized inference
evolved_{instruct,thinking}.py    best programs found by evolution
best_{accuracy,speed,overall}.py  selected on a single-use reserve slice
evolve_{instruct,thinking}.py     entry points of the evolution system
alphaevolve/                      database, prompt sampler, LLM client, diff applier, static checks, cascade, loop
route_quant.py                    the two-model router (Finding 2)
evaluate.py                       CharXiv evaluator with split support (dev / evo / holdout / reserve)
scores/                           every scores file behind every reported number, with per-answer responses
population/                       full evolution runs: candidates, per-candidate scores, LLM records, checkpoints
RESULTS.md, docs/REPORT_NOTES.md  lab notebook and overfitting audit
report/, presentation/            paper and slides
```

## How to run

> **Requirements:** a CUDA GPU (reference: RTX 3090 24 GB) for any evaluation. For re-running evolution you also need
> an API key for the mutation LLM. The client in `alphaevolve/llm.py` targets ArvanCloud's OpenAI-compatible gateway
> to Gemini (`ARVAN_API_KEY` in a git-ignored `.env`). To use Google's Gemini API or any other OpenAI-compatible
> endpoint, change `BASE_URL`, the auth header and the model names in that file.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_images.py          # 2,323 CharXiv images
python -m pytest -q tests/                 # GPU-free tests (extraction, database, loop, API failure handling)

python evaluate.py manual_instruct                        # 128 dev queries, ~1 min
python evaluate.py evolved_instruct --split holdout       # 800 holdout queries

bash reproduce.sh              # all programs on dev (~2–3 h, dominated by the Thinking model)
bash reproduce.sh --holdout    # + the holdout table (~14 h more)
bash reproduce.sh --evolve     # + both evolution runs (needs the API key, ~5 h)

python evolve_instruct.py --no-push        # a new evolution run without pushing checkpoints to git
```

Without a GPU you can still inspect every result. Each scores file in `scores/` stores the raw and post-processed
response for every query, and `scripts/replay_postprocess.py` re-scores post-processing changes offline.

## Credits

The CharXiv benchmark, its data and the evaluation harness under `charxiv/` are by Wang et al. (Princeton NLP), and the
starting script and evaluator scaffold came with the original task. AlphaEvolve is by Novikov et al. (Google
DeepMind). The models are Qwen3-VL-2B-Instruct and -Thinking by the Qwen Team. Built with AI coding assistance
(Claude Code).
