# Results log

Every significant run, newest last. Numbers come from `scores/<split>/scores_<id>.json`; the file is the source
of truth if this log and a file disagree.

- **Reference GPU: RTX 3090** since 2026-10-14, when the Colab workflow was retired. All speed comparisons in the
  report are on the 3090. The earlier T4 (FP16) numbers are kept below as history and are not compared with 3090 times.
- **Splits and their status** (see `CONTRACT.md` §2):
  - `dev` 128 queries: **design** set for manual work (rules and prompts are written from its failures); optimistic.
  - `reserve[:400]`: **partial tuning set** for manual Instruct since step 2a–2c used it for keep/drop decisions;
    still less biased than dev, but no longer clean.
  - `evo` 400: evolution fitness only.
  - `holdout` 800: **the only clean test** for the final comparison of all versions.

  The dev − reserve gap of manual Instruct grew from 4.5 points (step 1: 56.2 vs. 51.7) to 10.4 points (step 2c:
  65.6 vs. 55.2), so dev numbers of manual versions overstate generalization.
- **Time** is `total_time` from `evaluate.py`, which includes the model load on the first query.

## Summary

Generated from `scores/` by `python scripts/results_table.py --write`; do not edit between the markers. Milestone
runs use step-specific ids (e.g. `manual_instruct_s2b`) so that every step keeps its row.

<!-- results-table:start -->
| Date | Id | Model | Split | GPU | Accuracy | Total time | s/query | First query | Errors |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14 | `baseline_starting_scripts` | Instruct | dev | T4 | **38.3%** (49/128) | 329 s | 2.57 | 18.5 s | 0 |
| 2026-10-14 | `baseline_starting_scripts_3090` | Instruct | dev | RTX 3090 | **38.3%** (49/128) | 250 s | 1.95 | 15.0 s | 0 |
| 2026-10-14 | `baseline_starting_scripts_thinking` | Thinking | dev | RTX 3090 | **0.0%** (0/128) | 805 s | 6.29 | 93.2 s | 0 |
| 2026-10-14 | `manual_instruct` | Instruct | dev | RTX 3090 | **56.2%** (72/128) | 267 s | 2.09 | 15.8 s | 0 |
| 2026-10-14 | `manual_instruct_s2b` | Instruct | dev | RTX 3090 | **60.2%** (77/128) | 67 s | 0.52 | 12.9 s | 0 |
| 2026-10-14 | `manual_instruct_s2c` | Instruct | dev | RTX 3090 | **65.6%** (84/128) | 72 s | 0.57 | 11.4 s | 0 |
| 2026-10-14 | `manual_thinking_s1` | Thinking | dev | RTX 3090 | **72.7%** (93/128) | 2849 s | 22.26 | 32.9 s | 0 |
| 2026-10-16 | `best_accuracy` | Thinking | dev | RTX 3090 | **74.2%** (95/128) | 1985 s | 15.51 | 31.1 s | 0 |
| 2026-10-16 | `best_overall` | Instruct | dev | RTX 3090 | **67.2%** (86/128) | 57 s | 0.45 | 10.8 s | 0 |
| 2026-10-16 | `best_speed` | Instruct | dev | RTX 3090 | **65.6%** (84/128) | 57 s | 0.44 | 9.9 s | 0 |
| 2026-10-16 | `evolved_instruct` | Instruct | dev | RTX 3090 | **67.2%** (86/128) | 59 s | 0.46 | 10.5 s | 0 |
| 2026-10-16 | `evolved_thinking` | Thinking | dev | RTX 3090 | **74.2%** (95/128) | 1979 s | 15.46 | 31.9 s | 0 |
| 2026-10-15 | `baseline_starting_scripts` | Instruct | holdout | RTX 3090 | **25.9%** (207/800) | 2087 s | 2.61 | 15.6 s | 0 |
| 2026-10-15 | `evolved_instruct` | Instruct | holdout | RTX 3090 | **62.0%** (496/800) | 368 s | 0.46 | 11.5 s | 0 |
| 2026-10-15 | `manual_instruct` | Instruct | holdout | RTX 3090 | **56.0%** (448/800) | 331 s | 0.41 | 12.6 s | 0 |
| 2026-10-15 | `manual_thinking_h000` | Thinking | holdout | RTX 3090 | **64.5%** (258/400) | 11699 s | 29.25 | 16.3 s | 0 |
| 2026-10-15 | `manual_thinking_h400` | Thinking | holdout | RTX 3090 | **65.0%** (260/400) | 11186 s | 27.97 | 65.1 s | 0 |
| 2026-10-16 | `evolved_thinking` | Thinking | holdout | RTX 3090 | **67.0%** (536/800) | 15419 s | 19.27 | 2506.9 s | 0 |
| 2026-10-16 | `route_quant_check32` | Instruct | holdout | RTX 3090 | **59.4%** (19/32) | 368 s | 11.50 | 11.2 s | 0 |
| 2026-10-14 | `baseline_starting_scripts` | Instruct | reserve | RTX 3090 | **32.0%** (128/400) | 892 s | 2.23 | 13.6 s | 0 |
| 2026-10-14 | `manual_instruct_s2b` | Instruct | reserve | RTX 3090 | **51.2%** (205/400) | 165 s | 0.41 | 11.2 s | 0 |
| 2026-10-14 | `manual_instruct_s2c` | Instruct | reserve | RTX 3090 | **55.2%** (221/400) | 177 s | 0.44 | 10.8 s | 0 |
| 2026-10-14 | `manual_thinking_s1` | Thinking | reserve | RTX 3090 | **59.4%** (76/128) | 3217 s | 25.13 | 16.7 s | 0 |
| 2026-10-15 | `manual_instruct_s2c_r400` | Instruct | reserve | RTX 3090 | **58.8%** (235/400) | 153 s | 0.38 | 10.2 s | 0 |
| 2026-10-15 | `route_quant_r400` | Instruct | reserve | RTX 3090 | **67.8%** (271/400) | 4287 s | 10.72 | 52.5 s | 0 |
| 2026-10-16 | `evolved_instruct_r800` | Instruct | reserve | RTX 3090 | **67.0%** (134/200) | 89 s | 0.45 | 13.3 s | 0 |
| 2026-10-16 | `evolved_thinking_r800` | Thinking | reserve | RTX 3090 | **72.0%** (144/200) | 2873 s | 14.37 | 19.0 s | 0 |
| 2026-10-16 | `manual_instruct_r800` | Instruct | reserve | RTX 3090 | **63.0%** (126/200) | 87 s | 0.44 | 11.3 s | 0 |
| 2026-10-16 | `manual_thinking_r800` | Thinking | reserve | RTX 3090 | **71.0%** (142/200) | 4343 s | 21.72 | 18.7 s | 0 |
<!-- results-table:end -->

**Offline replays** (post-processing applied to recorded raw responses; no scores file, no timing):

| Date | Program | Replayed on | Split | Accuracy |
|---|---|---|---|---|
| 2026-10-14 | `manual_instruct` step 1 | `baseline_starting_scripts_3090` | dev | **56.2 %** (72/128), equal to its GPU run |
| 2026-10-14 | `manual_instruct` step 1 | `baseline_starting_scripts` | reserve[:400] | **51.7 %** (207/400) |

**Tags:** `v0.1-baseline` (T4 baseline), `v0.1-baseline-3090`, `v0.2-baseline-thinking`, `v0.3-manual-instruct-seed`,
`v0.4-manual-instruct-extraction` (scores id `manual_instruct`).

> The Thinking baseline's first query includes the one-time model download (93 s); without it the run is about
> 730 s (712 s for queries 2–128 plus a ~15 s load). The Colab runs of session 1 never reached the repo and are superseded.

## Entries

### 2026-10-14: Instruct baseline (`starting_scripts.py`, unchanged)

- **Result:** 38.3 % on dev (README: about 37 %). Accuracy is within the expected range, so the setup (images,
  prompt, scoring, greedy decoding) is confirmed.
- **Per category:** enum 91 %, title 67 %, ocr 56 %, layout 56 %, trend 50 % (1/2), bool 40 %,
  **quant 2 % (1/49)**.
- **Main failure: verbose answers.** 52 responses are longer than 60 characters, and only 3 of them are correct.
  Those 52 take 254 s of the 310 s spent on queries (82 %). The other 75 short responses are 61 % correct and take 56 s.
  In 34 of the 79 wrong answers, the gold answer appears verbatim inside the response.
- **Other failures:**
  - "Not Applicable" questions: 5/32 correct; the model invents a value instead (qid 14/15).
  - Notation: `10²` vs gold `10^2`, LaTeX `$10^{-1}$` vs `10^-6`, `2 by 1` vs `1 by 2`.
  - Counting questions (qid 10, 12, 17, 19) reason step by step and run out of the 128-token limit.
- **Timing:** model load is about 18 s (first query); short answers take about 1 s/query.
- **Reference point:** answering "Not Applicable" to everything scores 25 % on every split.
- **Conclusion:** constraining the output format should improve accuracy and speed together. That is the first
  manual optimization, plus a fallback that extracts a short answer from a long response.

### 2026-10-14: `manual_instruct.py` step 1, answer extraction only (offline replay)

- **Change:** `postprocess()` is added to the seed (`v0.3-manual-instruct-seed`, identical to `starting_scripts.py`).
  Prompt, generation and model are unchanged. Responses that are prose (multi-line or sentence-like) are reduced to
  their final answer, tried in this order:
  1. an explicit `Answer: X` marker;
  2. a bare last line;
  3. the last sentence's "… is X".

  Bare answers pass through unchanged.
- **Method:** decoding is greedy, so raw outputs do not depend on post-processing. The result comes from replaying
  `postprocess` on the 128 recorded baseline responses (`scripts/replay_postprocess.py`). This isolates the effect of
  extraction exactly. A GPU run of the same commit should reproduce it.
- **Result (dev, offline): 38.3 % → 56.2 % (49 → 72; +23 fixed, 0 broken).** quant 1 → 14/49, ocr 24 → 34/43;
  other categories unchanged.
- **Not recoverable by extraction:** 17 prose responses stay wrong.
  - Truncated reasoning with no final answer: 7.
  - A wrong value stated explicitly: 10, mostly invented counts or values where the gold answer is "Not Applicable".
  - These need prompt or generation changes, not post-processing.
- **Caveat:** the rules were written while reading these dev responses, so 56.2 % is optimistic. The generalization
  check is a replay on raw responses from a `reserve` slice (see `CONTRACT.md` §2).
- **Time:** unchanged by construction (same generation). Speed comes in later steps.

### 2026-10-14: Instruct baseline on the RTX 3090 (new reference GPU)

- **Setup:** `starting_scripts.py` unchanged (FP16, as hard-coded in the script) on the rented RTX 3090.
  Scores: `scores/dev/scores_baseline_starting_scripts_3090.json`.
- **Result:** 38.3 % (49/128), the same as the T4. 9 of 128 responses differ in wording between the GPUs (same dtype,
  different kernels), and none changes correctness, so the T4 accuracy results carry over. Greedy outputs are
  deterministic per GPU type, not across GPU types: compare responses only between runs on the same GPU.
- **Time:** 250 s total, 1.95 s/query, first query (including model load) 15.0 s. The T4 took 329 s.
- **Conclusion:** the 3090 becomes the reference GPU; this run is the speed baseline for everything that follows.

### 2026-10-14: `manual_instruct.py` step 1 confirmed on GPU (3090)

- **Result: 56.2 % (72/128), exactly the offline replay.** Replaying `postprocess` on the 3090 baseline responses also
  gives 72/128 (same per-category numbers as on the T4 responses), and all 128 GPU responses equal the replayed ones.
  Scores: `scores/dev/scores_manual_instruct.json`.
- **Time:** 267 s vs. 250 s for the baseline (+7 %). Generation is unchanged, so this is timing noise, as expected.
- **Conclusion:** extraction works as measured offline; tagged `v0.4-manual-instruct-extraction`. The dev number is
  still optimistic (rules written from dev); the reserve replay gives the unbiased gain.

### 2026-10-14: Extraction generalizes: reserve replay (**reserve used:** first 400 queries)

- **Reserve use:** `starting_scripts.py` was run on `reserve --n 400` (the first 400 reserve queries, 100 figures) to get
  raw responses that the extraction rules were not written from. Scores:
  `scores/reserve/scores_baseline_starting_scripts.json`. This slice is now *used* for post-processing evaluation;
  later fresh checks take reserve queries after the first 400.
- **Baseline on this slice:** 32.0 % (128/400), lower than dev (38.3 %): this slice has more quant questions
  (165/400 = 41 %, vs. 38 % on dev), where the baseline scores 3/165.
- **Replay of `manual_instruct` step 1: 32.0 % → 51.7 % (+19.7 points; 79 fixed, 0 broken).** On dev the gain was
  +17.9 points, so the rules are not overfit to dev. By category: quant 3 → 48/165, ocr 65 → 95/137, layout 16 → 20/27;
  the rest unchanged.
- **Still wrong after extraction (63 of 142 changed responses):**
  - 29: wrong value or truncated reasoning with no final answer;
  - 21: the gold answer appears somewhere in the response but a different span is extracted (e.g. "a total of 5
    labeled ticks"); some of these are loose substring matches like "0" in "20";
  - 13: gold is "Not Applicable" and the model invents a value.

  As on dev, most need prompt or generation changes (step 2), not more extraction rules. The rules are **not** tuned
  on these reserve failures, which would spend the slice.
- **Time:** 892 s for 400 queries (2.23 s/query). The 192 responses longer than 60 characters take 787 s (88 %),
  and only 1 of them is correct before extraction (dev: 82 %).
- **Conclusion:** extraction stays as the safety net. Next is step 2: an answer-only prompt plus a smaller
  `max_new_tokens`, measured on dev with a GPU run.

### 2026-10-14: Thinking baseline (`baselines/starting_scripts_thinking.py`, model name changed only)

- **Result: 0.0 % (0/128)**, as expected. With `max_new_tokens=128`, 105 of 128 responses stop inside the reasoning.
  The other 23 do reach `</think>`, but the evaluator compares the whole response (reasoning included) with the gold
  answer, so even those score 0.
- **The model is accurate when it finishes:** the text after `</think>` is correct for **20 of those 23** (qids 1–5, 7,
  11, 13, 18, 19, i.e. the short lookup questions). This was computed from the recorded responses and is not a score
  of any program.
- **Time:** 805 s total, but the first query (93 s) includes downloading the model, which happened during this run.
  Queries 2–128 take 712 s (median 5.7 s/query, max 7.2 s): every query generates close to the full 128 tokens.
- **Conclusion for `manual_thinking.py`:** (1) return only the text after `</think>`, with the step-1 extraction as a
  fallback; (2) raise the reasoning budget well above 128 tokens and, when it runs out, force `</think>` and generate a
  short answer. The budget is the main accuracy–speed trade-off: at roughly 22 tokens/s, every 128 extra tokens cost
  about 6 s/query. The next measurement is the distribution of reasoning lengths with a large budget.

### 2026-10-14: `manual_instruct.py` step 2a, extraction robustness (offline replay, dev and reserve[:400])

- **Change (extraction only; prompt and generation unchanged):** two rules, written from the 18 dev responses that
  step 1 changed but left wrong (not from reserve failures):
  1. **cut-off tail:** a response ending in an unfinished sentence ("…\n\nThere are", "Therefore, the", "The
     spatially", an open `$10^{-`) has that fragment dropped before extraction;
  2. **notation:** Unicode superscripts become caret notation as in the gold answers (`PPL (10²)` → `PPL (10^2)`),
     for bare answers too.
- **Result (replay, exact since generation is unchanged):**

  | | dev (3090 raw) | reserve[:400] |
  |---|---|---|
  | step 1 | 72 | 207 |
  | + cut-off tail | 72 | 207 |
  | + notation | 73 | 207 |
  | both (step 2a) | **73 (57.0 %)** | **207 (51.7 %)** |

  0 broken on either split.
- **Negative result (report):** more extraction rules did not help. Both rules were plausible from dev failures, but
  only one dev answer improved and none on reserve.
- **Conclusion: extraction is saturated on baseline-style outputs.** The cut-off rule changes 22 more reserve
  responses but fixes none: in a truncated response the last complete sentence is usually a list item, not the
  answer. The remaining errors need generation changes (step 2b). The cut-off rule is kept for now because step 2b
  uses a smaller `max_new_tokens`; it is re-ablated on step 2b's recorded raw outputs and removed if it still adds
  nothing. No GPU run: the replay was shown to equal the GPU run response by response (step 1).

### 2026-10-14: `manual_instruct.py` step 2b, answer-only prompt + `max_new_tokens` 64 (GPU, reserve then dev)

- **Change (one, on top of step 2a):** after the question, append "Answer with only the final answer, written exactly
  as in the chart (a number, word or short phrase). Do not explain." and lower `max_new_tokens` from 128 to 64 (the
  longest dev gold answer is 30 tokens). Extraction (step 2a) stays as the safety net. Scores id `manual_instruct_s2b`.
- **Results:**

  | | reserve[:400] acc. | reserve time | dev acc. | dev time |
  |---|---|---|---|---|
  | step 2a (raw from baseline + extraction) | 51.7 % (207) | 892 s (2.23 s/q) | 57.0 % (73) | 250–267 s |
  | **step 2b** | **51.2 % (205)** | **165 s (0.41 s/q)** | **60.2 % (77)** | **67 s (0.52 s/q)** |
  | step 2b raw, no extraction | 51.2 % (205) | | 58.6 % (75) | |

- **Accuracy:** flat on reserve (−2, within noise; SE ≈ ±2.5 points), +4 on dev. Dev is the split the prompt could
  overfit to, and the honest reserve number says the prompt gives the same accuracy as prose + extraction.
  By category on reserve, vs. step 2a: layout +4, ocr +1, enum −2, title −2, quant unchanged at 48/165.
- **Speed: 5.4× faster on reserve, 4× on dev.** Only 5 of 400 raw responses are longer than 60 characters. GPU
  utilization during the run is about 11 %, so per-query time is now dominated by CPU-side preprocessing (slow
  image processor, `use_fast=False`): the first target for later speed work.
- **Extraction after step 2b:** changes 2 dev responses (both fixes, notation rule) and 1 reserve response (still wrong). The cut-off
  rule from step 2a touches **no** response on either split, so it is removed (next commit; outputs unchanged).
- **"Not Applicable" is the largest remaining error:** the model answers it 20 times on reserve, always correctly,
  but misses 80 of the 100 gold "Not Applicable" (dev: 6 of 32). Precision 100 %, recall 20 %. Step 2c targets this.
- **Conclusion:** keep. Same accuracy as step 2a at a fifth of the time; tagged `v0.6-manual-instruct-answer-only`.

### 2026-10-14: `manual_instruct.py` step 2c, "Not Applicable" check in the prompt (GPU, reserve then dev)

- **Design (from dev only):** on dev after step 2b, 26 of 32 gold "Not Applicable" answers were missed, 15 of them on
  the colorbar questions (qid 14/15), where the model reads axis ticks as a colorbar. Answering "Not Applicable" for
  whole question types would exploit the base rate (the overfitting warning in `CONTRACT.md` §4), so the change is
  prompt guidance only.
- **Change (one, on top of step 2b):** for questions that contain a "Not Applicable" condition, append: *First check
  the condition for "Not Applicable" in the question (for example, whether the plot really has a colorbar, lines or
  the element asked about); if it holds, answer exactly "Not Applicable".* Scores id `manual_instruct_s2c`.
- **Results:**

  | | reserve[:400] | dev |
  |---|---|---|
  | step 2b | 51.2 % (205/400), 165 s | 60.2 % (77/128), 67 s |
  | **step 2c** | **55.2 % (221/400), 177 s** | **65.6 % (84/128), 72 s** |
  | "Not Applicable" recall | 20 → 44 / 100 | 6 → 15 / 32 |
  | "Not Applicable" precision | 100 % → 73 % (44/60) | 100 % → 75 % (15/20) |
  | correct on non-NA questions | 185 → 177 | 71 → 69 |

- **Reading:** +24 NA answers gained against 8 others lost on reserve, net +16 (+4.0 points). The model answers
  "Not Applicable" for 15 % of reserve queries, below the 25 % base rate, so this is not collapse toward the constant.
  By category on reserve vs. step 2b: quant 48 → 60, bool 7 → 10, title 18 → 21, ocr 96 → 94. Time +7 %: noise.
- **Precision/recall finding (report):** a prompt nudge toward the abstain answer trades precision for recall. Of the
  9 reserve answers that step 2b had right and 2c gets wrong, only 4 became "Not Applicable" (qid 9, 13 ×2, 3); the
  other 5 are changed values (counts 4 → 3 on qid 12 ×2, qid 9, 10, 3), i.e. the longer prompt perturbs unrelated
  answers. All 16 false "Not Applicable" on reserve cluster by question type: qid 3 (y-axis label) 4, qid 13 (legend
  labels) 3, then 2 each for qid 8, 9, 14. Dev: 5 lost (1 to NA), false NA mostly qid 12. Per-question-type NA
  guidance is therefore a mutation dimension for evolution, not a manual fix (the clusters were seen on reserve).
- **Conclusion:** keep; tagged `v0.7-manual-instruct-na-check`. Remaining large error: quant (60/165 on reserve),
  i.e. counting and reading values, which an answer-only prompt cannot reason through.
- **Reserve status:** reserve[:400] has now been used for three step decisions (2a–2c), so it is a validation set for
  manual Instruct. The unbiased final number comes from `holdout` (or a fresh reserve slice after the first 400).

### 2026-10-14: Thinking reasoning-length probe (dev[:32], budget 3072) and `manual_thinking.py` step 1

- **Probe** (`scripts/think_probe.py`, data `analysis/think_probe_dev32.json`; `manual_thinking` step 1 code with
  `_THINK_BUDGET` raised to 3072): **75 % (24/32), 57 s/query**, about 23.5 tokens/s with ~25 % GPU utilization
  (per-token Python overhead of HF `generate`). Instruct step 2c gets 22/32 on the same queries (+2, not significant).
- **Reasoning lengths are bimodal:** 20/32 finish reasoning, all within 1024 tokens (median of all 32: 530 tokens;
  18 of the 20 correct); the other 12 never finish, second-guessing ("Wait, …") until the cap, and a forced answer
  gets 6 of them right. No response finishes between 1024 and 3072 tokens.

  | budget | responses that finish | correct among them |
  |---|---|---|
  | 256 | 12 | 11 |
  | 512 | 16 | 15 |
  | 1024 | 20 | 18 |
  | 3072 | 20 | 18 |

- **Step 1 (code at tag `v0.9-manual-thinking-step1`):** `_THINK_BUDGET = 1024` (loses no finishing response, cuts
  the worst case from ~130 s to ~45 s), answer = text after `</think>`, forced `</think>` + short answer when the
  budget runs out, prompt and extraction from `manual_instruct` step 2b/2c. Estimated ~26 s/query, so it is measured
  on reserve[:128] and dev instead of reserve[:400] (~3 h). Results: entry below.
- **Provenance:** the step-1 code was committed by mistake inside `1a1e9f6` (message about CONTRACT.md only); see the
  tag annotation and docs/HANDOFF.md §11.

### 2026-10-15: `manual_thinking.py` step 1 results (GPU, reserve[:128] then dev; run 2026-10-14 evening)

- **Setup:** code at tag `v0.9-manual-thinking-step1` (budget 1024). Scores id `manual_thinking_s1`. Compared with
  `manual_instruct` step 2c on the **same** queries (its scores restricted to reserve[:128]).
- **Results:**

  | | reserve[:128] | dev (128) |
  |---|---|---|
  | **Thinking step 1** | **59.4 % (76)**, 3,217 s, 25.1 s/q | **72.7 % (93)**, 2,849 s, 22.3 s/q |
  | Instruct step 2c, same queries | 57.0 % (73), ~0.44 s/q | 65.6 % (84), 0.56 s/q |
  | only Thinking right / only Instruct right | 17 / 14 | 18 / 9 |

  Reserve difference +3 (within noise, SE ≈ ±4 points); dev +9 (dev is optimistic for both). About 50× slower.
- **Budget exhaustion is the cost driver:** 46 of 128 responses on each split hit the 1024-token budget and get a
  forced answer. They take **65–69 % of the time** (42–46 s each vs. 11–14 s for finished ones) and are right only
  30 % (reserve, 14/46) / 41 % (dev, 19/46), vs. 76 % / 90 % for responses that finish.
- **Where Thinking helps: quant.** reserve 24 vs. 16 /51, dev 32 vs. 21 /49 (counting and value reading benefit from
  reasoning). It is worse on enum (1 vs. 4 /8; 6 vs. 9 /11) and layout (3 vs. 6 /8 on reserve); ocr and title equal.
- **NA:** recall 14/32 (reserve), 15/32 (dev), precision 14/15 and 15/16: Thinking abstains less but more precisely
  than Instruct step 2c (73 %).
- **Extraction** changes 1 answer (dev): the text after `</think>` is almost always already bare.
- **Conclusion:** a defensible Thinking baseline: slightly more accurate than Instruct overall, clearly better on
  quant, at ~50× the time, most of it spent on responses that never finish reasoning. The two models are
  complementary (17 + 14 disagreements on reserve), which suggests routing quant questions to Thinking (see
  docs/HANDOFF.md §10). Code tagged `v0.9-manual-thinking-step1`.

### 2026-10-15: The "Wait, …" pattern in Thinking step 1 (measurement; `scripts/wait_probe.py`)

- **Frequency** (recorded `manual_thinking_s1` reasoning): "Wait" appears in **76 % of reserve[:128] and 77 % of dev
  responses**; median 7 (reserve) / 4 (dev) per response. Responses that exhaust the budget average 22–24 "Wait"s
  (forced answer, 30–41 % correct); finished ones average 3–4 (76–90 % correct). In the 3072-token probe, responses
  that never finish reach a mean of 86. The first "Wait" comes early (median ~65 tokens into the reasoning).
- **Does the answer change after it? Intervention:** cut the recorded reasoning just before the k-th "Wait", force
  `</think>` + answer exactly as `manual_thinking` does at its budget, compare with the recorded answer (responses
  with fewer than k "Wait"s keep their answer). Text-based heuristics for "the answer before the first Wait" were
  tried first and rejected: the text there is usually not an answer yet.

  | cut before | reserve[:128] acc. | right→wrong / wrong→right | dev acc. | right→wrong / wrong→right | mean reasoning tokens (res. / dev) |
  |---|---|---|---|---|---|
  | none (recorded, budget 1024) | 76 | | 93 | | 566 / 540 |
  | Wait #1 | 76 | 9 / 9 | 88 | 13 / 8 | 88 / 83 |
  | Wait #2 | 78 | 5 / 7 | 92 | 8 / 7 | 141 / 134 |
  | Wait #4 | **84** | 2 / 10 | **93** | 8 / 8 | **215 / 220** |

- **Reading:** second-guessing changes many answers in both directions, but on net it does not help: stopping before
  the 4th "Wait" keeps dev accuracy (93 → 93) and raises reserve (76 → 84, +6 points, above the ±4 noise level),
  with **~60 % fewer reasoning tokens**. At ~23.5 tokens/s that is roughly 25 → 10 s/query (estimate; not a timed run).
  Even stopping at the first "Wait" costs little (reserve ±0, dev −5) with 85 % fewer tokens.
- **Caveats:** a re-tokenized text prefix, not the original token ids; one forced-answer phrasing; this measures a
  cut of *recorded* reasoning, not a program run end to end.
- **Use (user decision 2026-10-14):** not applied manually. Bounding second-guessing is a mutation direction for
  `evolve_thinking.py`; this measurement is reported as the motivation and as a reference point for what evolution finds.

### 2026-10-15: Instruct evolution pilot `instruct_20261015-0904` (machinery check; branch `evo/instruct_20261015-0904`)

- **Setup:** seed `manual_instruct.py` (step 2c), 5 generations × 1 child, 3 islands, cascade 8 → 64 → 400 evo
  queries, mutation LLM `Gemini-3.1-Flash-Lite-Preview` (ArvanCloud), inspirations as diffs. Run in three phases to
  test resume: A = new run to generation 2 (clean stop); B = resume, process killed during generation 4's subset
  stage (after its LLM call and smoke stage); C = resume to generation 5.
- **Machinery: everything worked.** Seed evaluation, sampling, prompts, LLM calls, diff application, static checks,
  cascade, database (islands rotate per child, migration at generation 3, MAP-Elites cells by time × quant accuracy),
  atomic checkpoints, one commit per generation pushed to `evo/<run_id>`. On resume after the kill, generation 4 was
  redone with the recorded LLM response (file unchanged, no new call) and its finished smoke stage was skipped.
- **Results (evo, 400 queries; fitness only, not a generalization number):**

  | id | parent | change | full evo |
  |---|---|---|---|
  | seed | | manual_instruct step 2c | 54.7 % (219) |
  | g001_c00 | seed | cache processed images per path | 54.7 % (identical answers) |
  | g002_c00 | seed | same caching idea again | 54.7 % (identical answers) |
  | g003_c00 | seed | system prompt "precise chart-reading assistant, only the final exact answer" | **55.7 % (223)** |
  | g004_c00 | g003 | caching + "N by M" hint for layout questions | 55.5 % |
  | g005_c00 | g001 | adopts g003's system prompt from an inspiration (crossover) | 55.7 % |

  g003's +4 answers (quant +5, layout +1, enum +1, ocr −2, title −1) are within noise (SE ≈ ±2.5 points).
  Image caching saves 1–4 % time: below the ~10 % noise level.
- **Cost:** 5 LLM calls, $0.0105 total (≈ 3.3k input / 0.2–1.2k output tokens per call, no thinking tokens).
  GPU: ~3.5 min per child that reaches the full stage (three model loads plus 472 queries).
- **Issues found (to fix before the full run):**
  1. LLM usage in the checkpoint undercounts after a crash: the call made after the last checkpoint is not counted
     on resume (checkpoint says 4 calls / $0.0077; the `llm/` records show 5 / $0.0105).
  2. Ties in accuracy are broken by raw time, so timing noise picks the "best" program (g001 over the seed,
     g005 over g003 on 1–4 % differences).
  3. Low diversity from the Lite model: two of five children were the same caching idea, once against the
     sampled directions; exact-hash dedupe does not catch near-duplicates.
  4. Cosmetic: the seed's `island` field records only the last island it was added to.

### 2026-10-15: Routing validation, `route_quant.py` vs. Instruct alone (**reserve used:** queries 400–800)

- **Reserve use:** reserve queries 400–800 (`--offset 400 --n 400`; figures disjoint from reserve[:400]), untouched
  until now, used once for this check. The design came from reserve[:128] and dev.
- **Programs:** `route_quant.py` sends number-reading/counting questions (qids 8, 9, 10, 12, 14, 15, 17, 19,
  recognized from the question text; 0 misrouted out of all 4,000 distinct CharXiv validation questions — an
  earlier "7,872" double-counted, see the overfitting audit in docs/REPORT_NOTES.md) to `manual_thinking` step 1 and the
  rest to `manual_instruct` step 2c. Baseline: `manual_instruct` step 2c alone on the same 400 queries.
- **Results:**

  | | accuracy | total time | s/query |
  |---|---|---|---|
  | Instruct alone (`manual_instruct_s2c_r400`) | 58.8 % (235/400) | 153 s | 0.38 |
  | **Routed (`route_quant_r400`)** | **67.8 % (271/400)** | 4,287 s | 10.72 |

  **+9.0 points** (+36 net; 49 answers right only when routed, 13 right only with Instruct; McNemar χ² ≈ 21,
  p < 0.001). The reserve[:128] estimate was 63 %; the gain held and was larger here.
- **Where it comes from:** all of it is the 164 routed questions: 75 → 111 correct. The 236 others give identical
  answers in both runs (same model, greedy). Per qid (Instruct → Thinking, of n): 8: 18 → 24/27, 9: 6 → 8/11,
  10: 12 → 15/18, 12: 14 → 19/23, 14: 11 → 13/21, 15: 5 → 14/36, 17: 2 → 12/21, 19: 7 → 6/7. Largest gains on
  counting ticks (17) and colorbar maximum (15, mostly correct "Not Applicable"); number of subplots (19) loses one.
- **Cost:** 28× slower than Instruct alone: the routed questions take 25.5 s each (4,181 s of 4,287 s). 55 of 164
  exhaust the 1,024-token budget (22 correct); finished ones are 89/109 correct.
- **Conclusion:** routing is the most accurate approach so far → candidate for `best_accuracy.py`, to be flagged in
  the report as a combined (two-model) approach. Not a `best_overall` candidate at this speed: with the overall score
  of the evolution config (halving time worth 5 points), Instruct alone scores far higher. If Thinking evolution
  bounds second-guessing (the "Wait" measurement suggests ~60 % fewer reasoning tokens), the routed time would drop
  roughly to ~4 s/query. Final number: holdout, together with every other version.

### 2026-10-15: Full Instruct evolution: first attempt abandoned, restarted as `instruct_20261015-1207`

- **`instruct_20261015-1131` (abandoned after generation 2; branch kept as a record):** the first 11 children showed
  (1) repeated ideas for the same parent (list-spacing normalization 4×, system prompt 3×; the near-duplicate filter
  caught 3 before any GPU time) and (2) every `Gemini-3-Flash-Preview` call wasted: the model thinks by default and
  ran out of output tokens (3,929 of 4,096 reasoning), so its edit was cut off. Best by then: 55.5 % (seed 54.7 %).
- **Fixes (commits `153d5d0`, `026e0a0`):** Flash Preview with `reasoning_effort` low and `max_tokens` 8192 (same
  prompt → complete edit, 468 tokens, $0.003); the prompt lists the edits already tried from the parent with their
  outcomes. User decision: restart for one consistent configuration (~35 min lost).
- **`instruct_20261015-1207`:** the full run (12 generations × 6 children), started 12:07 UTC from commit `026e0a0`.
  Results: entry to follow.

### 2026-10-15: Manual versions frozen (tag `v1.1-manual-frozen`)

- **Final manual deliverables:** `manual_instruct.py` = step 2c (last change `1695e17`), `manual_thinking.py` =
  step 1 (last change `1a1e9f6`); sha256 in the tag annotation. **No further changes until the report is written**
  (user decision). Freezing them allows their holdout runs (CONTRACT.md §2: holdout only for finished versions).
- **Holdout plan:** manual versions overnight after Thinking evolution (manual_thinking in two 400-query chunks,
  then manual_instruct, then the Instruct baseline); evolved versions the next morning; routing and best_* composed
  from those runs where possible. The Thinking naive baseline is not run on holdout (0 % by construction, 0/128 dev).

### 2026-10-15: Full Instruct evolution `instruct_20261015-1207` (12 generations × 6 children; evo = training split)

- **Setup:** seed `manual_instruct.py` (step 2c); cascade 8 → 64 → 400 evo queries, margin 0.03; ensemble 85 %
  Gemini-3.1-Flash-Lite-Preview / 15 % Gemini-3-Flash-Preview (reasoning low); "already tried" list in the prompt.
  12:07–14:53 UTC (2 h 46 min); branch `evo/instruct_20261015-1207`.
- **Results on evo (400 queries) — training fitness, not generalization:**

  | best for | program | evo accuracy | vs. seed (54.75 %) | s/query |
  |---|---|---|---|---|
  | accuracy | `g010_c01` | **62.50 %** | **+31 answers (+7.75 points)** | 0.385 |
  | overall | `g008_c02` | 62.25 % | +30 | 0.392 |
  | speed | `g006_c04` | 60.50 % | +23 | 0.369 |

  Best per generation: 54.7 (seed) → 57.3 (gen 1) → 57.5 (2–4) → 60.5 (6) → 62.3 (8) → 62.5 (10), then a plateau.
  By category (seed → `g010_c01`): quant 60 → 85, layout 18 → 21, ocr 93 → 96, enum 14 → 15, bool 11 → 12,
  title 21 → 19, trend 2 → 2. Speed unchanged (the run improved accuracy only).
- **Two independent lineages converge:** seed → g006_c04 → g007_c01 → g009_c01 → **g010_c01** (62.50 %) and
  seed → g001_c02 → g004_c02 → g007_c05 → **g008_c02** (62.25 %).
- **What changed along the way:** a system prompt ("precise chart-reading assistant", later refined to read ticks,
  legends and grids carefully), broader "Not Applicable" checking (lines, intersections, axes, not only colorbars),
  answer cleanup applied to every answer, image caching. Not yet ablated.
- **Children (72):** 38 full evaluation, 10 stopped at the 64-query stage, 6 duplicates, 4 near-duplicates,
  10 invalid edits, 3 rejected (static or smoke), 1 error. 21 never touched the GPU.
- **Cost:** 72 LLM calls (66 Flash-Lite, 6 Flash), **$0.23**; 3 transient connection errors retried, no pauses.
- **Export:** `g010_c01` → `evolved_instruct.py` (pre-defined rule: best evo accuracy; `g008_c02` is one answer
  behind, i.e. equivalent within noise). Tag `v1.2-evolved-instruct`. Its holdout run is queued after the overnight
  holdout queue.
- **Note:** an 8-query smoke check of `evolved_instruct.py` (16 s) ran on the GPU while the Thinking evolution was
  evaluating its seed: a few seconds of contention in a ~70-minute evaluation (< 0.5 %), below the timing-noise level.

### 2026-10-15: Thinking evolution `thinking_20261015-1453` (2 generations × 3 children; evo[:128] = training split)

- **Setup:** seed `manual_thinking.py` (step 1, frozen); cascade 4 → 32 → 128 evo queries, **margin 0** (speed rule
  unchanged); every prompt leads with bounding the reasoning; same ensemble and "already tried" list as Instruct.
  14:53–16:41 UTC; branch `evo/thinking_20261015-1453`; 6 LLM calls, $0.013.
- **Results (evo[:128], training fitness):**

  | program | edit | accuracy | s/query | status |
  |---|---|---|---|---|
  | seed | manual_thinking step 1 | 71.9 % (92) | 19.9 | |
  | **`g001_c00`** (Gemini-3-Flash) | `_THINK_BUDGET` 1024 → 512, comma spacing in lists | 74.2 % (95) | **14.0 (−30 %)** | full |
  | `g001_c02` | answer normalization | 53.1 % on the 32-query subset | 13.4 | stopped (margin 0) |
  | `g001_c01`, `g002_c00/01/02` | the same edit as `g001_c00` | — | — | exact duplicates (no GPU) |

- **Reading:** **−30 % time at the same accuracy** (+3 answers of 128 is within ±4 points; not claimed as a gain).
  Evolution chose the simplest suggested mutation (a lower budget); the "Wait"-stopping intervention measured earlier
  was stronger (~60 % fewer reasoning tokens with accuracy kept or raised). 4 of 6 children were byte-identical
  despite the "already tried" list. A faster child exists, so no third generation (user rule).
- **Next:** `g001_c00` → `evolved_thinking.py` (export after the holdout queues); its holdout run is queued tonight
  from the candidate file (`candidate_id` `evolved_thinking`, same code hash as the export).

### 2026-10-16: Holdout comparison (800 queries, RTX 3090) — the primary evidence

- **Runs** (overnight queues, 0 errors): `scores/holdout/scores_{baseline_starting_scripts, manual_instruct,
  evolved_instruct, manual_thinking_h000 + _h400, evolved_thinking}.json`. `evolved_thinking` was run from
  `population/thinking_20261015-1453/candidates/candidate_g001_c00.py` (same code hash as the export).
- **Routing is composed** from the Instruct and Thinking runs (Instruct answer for non-quant questions, Thinking answer
  for quant; time = sum of per-query latencies). **Check:** `route_quant` run directly on holdout[:32]
  (`scores_route_quant_check32.json`) gives 32/32 answers identical to the composition, 19/32 correct in both,
  time 368 s vs. 372 s composed.

  | program | accuracy | correct | s/query | bool /28 | enum /34 | layout /44 | ocr /305 | quant /326 | title /50 | trend /13 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | Instruct baseline | 25.9 % | 207 | 2.61 | 13 | 17 | 25 | 101 | 12 | 34 | 5 |
  | manual_instruct | 56.0 % | 448 | 0.41 | 15 | 17 | 33 | 213 | 132 | 34 | 4 |
  | **evolved_instruct** | **62.0 %** | 496 | 0.46 | 18 | 21 | 32 | 210 | 174 | 37 | 4 |
  | manual_thinking | 64.8 % | 518 | 28.61 | 18 | 12 | 38 | 232 | 177 | 34 | 7 |
  | **evolved_thinking** | **67.0 %** | 536 | **19.27** | 18 | 23 | 38 | 231 | 185 | 34 | 7 |
  | routing, manual models (composed) | 61.6 % | 493 | 14.45 | 15 | 17 | 33 | 213 | 177 | 34 | 4 |
  | routing, evolved models (composed) | 63.4 % | 507 | 8.03 | 18 | 21 | 32 | 210 | 185 | 37 | 4 |

- **Paired tests (McNemar, continuity-corrected):** evolved vs. manual Instruct 108 vs. 60, p = 0.0003; evolved vs.
  manual Thinking 31 vs. 13, p = 0.01; manual routing vs. manual Instruct 71 vs. 26, p < 1e-5; evolved vs. manual
  routing 43 vs. 29, p = 0.13 (n.s.); manual Thinking vs. manual Instruct 132 vs. 62, p < 1e-6; manual Instruct vs.
  baseline 248 vs. 7, p < 1e-50.
- **Findings:** Instruct evolution transfers (+7.75 on evo → **+6.0 on holdout**, from quant 132 → 174). Thinking
  evolution holds up (**−33 % time, +2.2 points, p = 0.01**). **Routing reversal:** Thinking alone beats routing
  (64.8 vs. 61.6 %) because Thinking is also better outside quant (ocr +19, layout +5); routing was validated only
  against Instruct alone. Routing remains +5.6 over Instruct and 2× faster than Thinking alone.
- **Selection consequences (user decision 2026-10-16):** routing was the pre-registered `best_accuracy` candidate;
  since holdout contradicted it, `best_accuracy` and `best_speed` are chosen on a fresh slice, reserve queries
  800–1000, to keep holdout clean. `best_overall` = `evolved_instruct`, decided on holdout; the margin over every
  alternative under the overall score makes the selection bias negligible.

### 2026-10-16: Selection of `best_accuracy` / `best_speed` on reserve queries 800–1000 (**reserve used**: 800–1000)

- **Slice:** reserve queries 800–1000 (`--offset 800 --n 200`), untouched until now, used once for this selection
  (user decision: holdout must not choose `best_accuracy` after it contradicted the pre-registered routing).
  `manual_thinking` was kept: without it, routing with the manual models cannot be measured on this slice.

  | candidate | accuracy (200) | s/query |
  |---|---|---|
  | **evolved_thinking** | **72.0 %** (144) | 14.37 |
  | manual_thinking | 71.0 % (142) | 21.72 |
  | routing, manual models (composed) | 70.0 % (140) | 9.87 |
  | routing, evolved models (composed) | 69.5 % (139) | 6.48 |
  | evolved_instruct | 67.0 % (134) | 0.45 |
  | manual_instruct | 63.0 % (126) | 0.44 |

  None of the top differences is significant on 200 queries (evolved_thinking vs. evolved routing 14 vs. 9,
  p = 0.4; vs. manual_thinking 8 vs. 6, p = 0.8). The ordering agrees with holdout's.
- **Selections (pre-defined rules):**
  - `best_accuracy.py` = **evolved_thinking** (most accurate on the slice). Its holdout number, **67.0 %**, is clean:
    holdout did not take part in the choice.
  - `best_speed.py` = **manual_instruct** (rule: unless evolved_instruct is > 10 % faster; 0.446 vs. 0.436 s/query on
    the slice, within noise; on holdout manual_instruct is 12 % faster). It gives up 4 points of accuracy to
    evolved_instruct for a speed difference that is noise: a consequence of the strict rule, stated as such.
  - `best_overall.py` = **evolved_instruct**, decided on holdout (62.0 % at 0.46 s/query; every more accurate program
    is 17–62× slower). The margin makes the selection bias negligible.
- **Files:** `evolved_thinking.py` = `g001_c00` (identical bytes; same code hash as its holdout run); `best_*.py` are
  copies of the selected programs with a two-line header comment. `reproduce.sh` added.

### 2026-10-16: `bash reproduce.sh` (dev) — reproduction and dev numbers of the evolved programs

- **Run:** 09:01–11:08 UTC on the RTX 3090, `EXIT=0`; GPU-free tests 60 passed. Log: `logs/reproduce_dev.log`.
- **Determinism:** baseline 49/128, manual_instruct 84/128, manual_thinking 93/128, each with **128/128 answers
  identical** to the committed dev runs.
- **New dev numbers** (scores copied to `scores/dev/scores_<program>.json`): evolved_instruct **67.2 %** (86/128,
  0.46 s/q), evolved_thinking **74.2 %** (95/128, 15.5 s/q); best_accuracy / best_overall / best_speed equal their
  source programs (95, 86, 84). Routing composed on dev: manual models 74.2 % (11.3 s/q), evolved models 75.0 %
  (7.2 s/q), i.e. at or above Thinking alone on dev, unlike on holdout (the routing reversal).
