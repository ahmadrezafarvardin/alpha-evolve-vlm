# Controller ↔ runner contract

The evolution **controller** proposes programs; the GPU **runner** scores them. They share only
folders of files, so the runner never needs an LLM API key and the controller's logic never needs a GPU.

```
controller                         candidates/candidate_<id>.py          ──►   runner (GPU)
  writes candidates, reads scores  scores/<split>/scores_<id>.json       ◄──   scripts/run_candidates.py
```

Both run on the same Colab VM (decided 2026-10-14): the controller calls the runner directly, and
git is used to save progress (§5). Everything below also works if the two are split across machines.

## 1. Candidates — `candidate_<id>.py`

- `<id>` matches `[A-Za-z0-9_-]+`. Evolution runs use `g<generation>_c<child>`, e.g. `g003_c02`;
  island and parent are recorded in the checkpoint, not the id. Ids are never reused: changed code gets a new id.
- A candidate is a self-contained module exposing
  `vlm_inference(image_path: str, question: str) -> str`, called once per query by `evaluate.py`.
  It may load the model lazily and cache it in module globals (as `starting_scripts.py` does).
- The evolvable region is delimited by `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END`.
- Rules (the controller rejects violations before writing the file; the runner does not check):
  - greedy decoding only (`do_sample=False`, or `temperature=0` for other backends);
  - only `Qwen/Qwen3-VL-2B-Instruct` or `Qwen/Qwen3-VL-2B-Thinking` (quantized variants allowed);
  - no reading `charxiv/data/` or any other source of gold answers.

## 2. Splits

Defined in `evaluate.py`: all from `charxiv/data/descriptive_val.json`, **disjoint by figure**, 4 questions per figure.

| Split | Figures | Queries | Use |
|---|---|---|---|
| `dev` | first 32 | 128 | the official set (`evaluate(program)` default); manual optimization iterates here |
| `evo` | 100 (seeded) | 400 | evolution fitness only; manual work never tunes on it |
| `holdout` | 200 (seeded) | 800 | final comparison of all versions; never used for tuning or selection |
| `reserve` | remaining 668 | 2,672 | labeled data kept untouched; see below |

Who may look at what:
- **Manual optimization** iterates on `dev`. Because rules are written while reading dev failures, dev accuracy
  of manual versions is optimistic. It is reported, but conclusions about generalization come from `holdout`.
- **Evolution** selects on `evo`, so evolved programs are optimistic on `evo`. Manual vs. evolved is therefore
  compared on **`holdout`**, which neither has seen; `evo` numbers are reported only as training fitness.
- **`holdout`** is evaluated only for finished versions (seeds, final manual, final evolved, best-* scripts).

**Why `reserve` exists.** The CharXiv test split has no public answers, so the 1,000 val figures are all the
labeled data we have. Once used for a decision, a split stops being an unbiased estimate. `reserve` keeps the
rest untouched for two purposes:
1. **A fresh check if `holdout` gets used up.** If holdout results end up influencing a decision (for example
   choosing between two final candidates), holdout becomes a validation set, and a slice of `reserve` gives the
   unbiased number for the report.
2. **Data for developing post-processing.** Raw responses on a reserve slice (e.g. `--split reserve --n 400`)
   can be used to test answer-extraction rules on outputs that were not used to write them. This keeps dev,
   evo and holdout clean.

Any use of `reserve` is logged in `RESULTS.md` with the slice used.

`evaluate.py` checks a fingerprint of the assignment and refuses to run if it changes (for example under a
different Python version), so split results are comparable across machines.

## 3. Runner — `python scripts/run_candidates.py [--split dev|evo|holdout] [--n N]`

For each `candidate_*.py` (sorted by name):

1. Skip it if `scores/<split>/scores_<id>.json` exists with the same `code_sha256`, `split` and `num_samples`
   (resume after a disconnect).
2. Run `python evaluate.py <candidate> --split <split> --out scores/<split>/scores_<id>.json` in a
   **fresh subprocess** with a timeout (default 3600 s). One candidate cannot crash, OOM or
   leak GPU memory into the next. The full stdout/stderr goes to `logs/<id>_<split>.log` (not committed).
3. If the subprocess fails, times out, or writes nothing, write a failure scores file (`status` ≠ `"ok"`).

Every candidate therefore ends with exactly one scores file per split it was run on. Scores files are written
atomically (`.json.tmp` then rename), so a reader never sees a half-written file.

## 4. Scores — `scores/<split>/scores_<id>.json` (schema version 1)

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | `1` |
| `candidate_id` | str | `<id>` |
| `status` | str | `"ok"`, `"error"` (crash, import failure, no output) or `"timeout"` |
| `program` | str | absolute path of the evaluated file on the runner |
| `code_sha256` | str | hash of the candidate file; the controller checks it matches what it wrote |
| `split` | str | `dev`, `evo`, `holdout` or `reserve` |
| `num_samples` | int | queries in this evaluation (N queries of the split, starting at `offset`) |
| `offset` | int | queries of the split skipped before these (0 = from the start; absent in older files means 0) |
| `accuracy` | float \| null | fraction correct, case-insensitive exact match after `.strip()`; `null` unless `status == "ok"` |
| `num_evaluated` | int | queries scored |
| `num_correct` | int | queries answered correctly |
| `num_errors` | int | queries where `vlm_inference` raised (scored as wrong) |
| `total_time` | float | seconds for all queries, **including the lazy model load on the first call** (same timer as the official `evaluate()`) |
| `avg_time_per_query` | float | `total_time / num_evaluated` |
| `first_query_time` | float | latency of the first query: approximately model load + one query |
| `per_qid` | object | `{"<qid>": {"correct", "total", "accuracy"}}` for each of the 19 question templates present |
| `per_category` | object | same, grouped as `title`, `ocr`, `quant`, `bool`, `enum`, `trend`, `layout` (CharXiv grading groups) |
| `responses` | list | every query: `key` (`<figure_id>_<subq_idx>`), `qid`, `category`, `response`, `gold`, `correct`, `latency`, and `raw_response` (model output before post-processing) when the program exposes a module global `last_raw_response`. Failure examples for the report and for mutation feedback come from here. |
| `wall_time` | float | seconds for the whole subprocess, including imports |
| `env` | object | `python`, `torch`, `transformers`, `gpu`, `hostname` |
| `timestamp` | str | UTC, ISO 8601 |
| `error` | object \| null | `null` if ok; otherwise `{"log_tail": "<last 4000 chars of output>"}` |

Fields from `num_evaluated` to `responses` are present only when `status == "ok"`.

Notes for the controller:
- **Compare speed only between runs on the same `env.gpu`.** The T4 is the reference GPU for the report.
- A program that answers `"Not Applicable"` to everything scores 25 % on every split (on dev: 100 % on qid 14,
  91 % on qid 15). Treat per-qid jumps toward such constants as a sign of overfitting, not progress.
- `responses[*].gold` is ground truth. Mutation prompts may show failures from `evo` only, never from
  `holdout`, `dev` or `reserve`.

## 5. Evolution runs and checkpoints — `population/<run_id>/`

`run_id` is `<model>_<YYYYMMDD-HHMM>`, e.g. `instruct_20261016-0930`. One folder holds everything about a run:

```
population/<run_id>/
├── config.json                      # written once: full run configuration + git commit of the code
├── checkpoint.json                  # rewritten after every generation (the resume point)
├── events.jsonl                     # append-only log, one JSON object per event
├── candidates/candidate_<id>.py     # every proposed program, including rejected and failed ones
├── scores/evo/scores_<id>[.n<N>].json  # runner output (§4), one per cascade stage reached (see below)
└── llm/<id>.json                    # mutation record: prompt, raw response, parsed diff, model, tokens, latency
```

**`config.json`**: model, seed program and its sha256, cascade stages and thresholds, island count and
migration interval, MAP-Elites feature definitions, LLM ensemble (models and sampling probabilities),
generation budget, random seed, `git_commit`.

**Evaluation cascade.** A full `evo` evaluation (400 queries) costs about 13 minutes on the 3090 for a
baseline-speed program, so every candidate goes through cheap stages first and only promising ones reach the full split:

| Stage | What runs | Cost | Passes if |
|---|---|---|---|
| 0. static | parse, imports, rules of §1 (greedy, allowed model, no gold data), dedupe by `code_sha256` | none | no violation, not a duplicate |
| 1. smoke | `evaluate.py --split evo --n 8` | ~30 s (mostly model load) | `status == "ok"`, `num_errors == 0`, responses not all identical |
| 2. subset | `evaluate.py --split evo --n 64` (first 64 evo queries = 16 figures) | ~2–3 min | promotion rule below |
| 3. full | `evaluate.py --split evo` (all 400) | ~13 min | always recorded; this is the fitness used for selection |

- **Promotion rule (stage 2 → 3).** With `a` = subset accuracy and `t` = subset `total_time`, a child is promoted if
  **either** holds:
  1. **not much worse:** `a_child ≥ a_parent − margin`, i.e. at most `margin` (default 0.03 = 3 points) *below* its
     parent; anything at or above the parent passes;
  2. **clearly faster at comparable accuracy:** `t_child ≤ 0.9 · t_parent` (at least 10 % faster) **and**
     `a_child ≥ a_parent − speed_margin` (default 0.06, about one standard error of 64 queries).

  64 queries have a standard error of about ±6 points, so `margin` is not zero; a strict rule would discard
  improvements on noise. Rule 2 lets speed-oriented mutations through without letting accuracy collapse. `margin`,
  `speed_margin` and the 10 % speed threshold live in `config.json`. The parent's subset score is its own stage-2
  result (seeds are run through stage 2 at the start of a run).
- **Same queries for everyone:** each stage uses the first N queries of `evo`, so all candidates at a stage are compared
  on identical inputs.
- **Fitness:** only stage-3 programs enter the MAP-Elites grid and the `best` pointers. Programs stopped at stage 1–2
  stay in `programs` with `stage_reached` and their partial scores, so the record is complete and the cascade's
  efficiency can be reported (fraction promoted, GPU time saved).
- **File names:** a partial stage writes `scores/evo/scores_<id>.n<N>.json` (e.g. `scores_g003_c02.n64.json`); the full
  stage writes `scores/evo/scores_<id>.json`. Ids never contain a dot, so the names cannot collide.
- Stages 1 and 2 are prefixes of stage 3's queries. Their results are not reused to shorten stage 3, because a fresh
  process is needed for honest timing (the model load is part of `total_time`).

**`checkpoint.json`** (`schema_version: 1`):

| Field | Meaning |
|---|---|
| `run_id`, `generation` | last **completed** generation |
| `programs` | `{id: {parent_id, island, generation, code_sha256, status, stage_reached, scores: {<split>: {accuracy, total_time, avg_time_per_query, per_category}}, features}}`. Rejected programs are kept, so the record is complete. |
| `islands` | `[[id, ...], ...]` members of each island |
| `map_elites` | `{island: {"<cell key>": id}}` elite per feature cell |
| `best` | `{accuracy: id, speed: id, overall: id}` |
| `rng_state` | Python `random` state, so selection continues the same random sequence |
| `llm_usage` | calls, input and output tokens, estimated cost in USD, calls per model; summed from `llm/*.json` (exact after a crash) |
| `timestamp` | UTC, ISO 8601 |

**Write order.** Within a generation, per child: `llm/<id>.json` (right after the call, so a crash never pays for it twice) → candidate file → scores → events. Only then is
`checkpoint.json` written, atomically (`.tmp` then rename). The checkpoint therefore never refers to a file
that does not exist yet, and a crash mid-write leaves the previous checkpoint intact.

**Resume** (`python evolve_instruct.py --resume <run_id>`): load `checkpoint.json` and redo generation
`generation + 1` from the start. Child ids depend only on (generation, slot), so the redone generation gets
the same ids. A child that already has `llm/<id>.json` reuses it instead of calling the LLM again, and the
runner skips any child already scored. A crash costs at most the GPU time of one unfinished evaluation.
A resumed run reloads the run's own `config.json` (including command-line overrides such as `--children`).
The model and temperature of each child are drawn from the random sequence even when its call is cached, so the
redone generation picks the same parents.

**Surviving the machine.** A local checkpoint alone does not protect a run if the machine is lost. After each
checkpoint the controller commits `population/<run_id>/` to the branch `evo/<run_id>` (with a temporary git index,
so `main`'s working tree and index are untouched) and pushes it, with a **120 s timeout**: a failed or hung push
only warns, and the next generation's push carries everything. To resume on a new machine: set it up
(docs/HANDOFF.md, Recovery), fetch `evo/<run_id>`, copy its `population/<run_id>/` into the working tree, run with
`--resume <run_id>`. At the end, the selected programs are copied to `evolved_instruct.py` / `evolved_thinking.py`
on `main` and the run folder is merged.

**Selection details.**
- *Timing noise never decides:* on equal accuracy a newer program replaces an elite or a `best` pointer only if it is
  more than 10 % faster (timing noise is ~7 %); the overall pointer needs more than the score change 10 % causes.
- *Near-duplicates:* before any GPU time, a child is rejected (`status: near_duplicate`) if its edit, as a set of
  changed lines with comments and layout ignored, has Jaccard similarity ≥ `near_duplicate_similarity` (0.8) with
  a sibling's edit (same parent). Children of different parents are never compared, so crossover is not blocked.
- Program statuses: `ok`, `stopped` (subset stage), `rejected` (static or smoke), `duplicate`, `near_duplicate`,
  `invalid_diff`, `llm_error`, `error`, `timeout`.

**Mutation LLM and API failures.** The ensemble member (model) is drawn per call; each `llm/<id>.json` records it.
- Transient errors (429, 5xx, 408/409, timeouts, connection) are retried with exponential backoff (10 s doubling,
  6 attempts). If they outlast the retries, the run **pauses** (e.g. an exhausted quota).
- 401 / 402 / 403 / 404 (bad key, no credit, forbidden, model retired or unknown) are confirmed by one retry after
  10 s (the gateway once returned a spurious 401), then the run **pauses**. 400 fails only that child (`llm_error`).
- Every failed attempt is appended to `population/<run_id>/api_errors.jsonl` (time, run_id, generation, child,
  model, status, kind, attempt).
- **Pause:** the checkpoint is *not* rewritten (it describes the last completed generation, with its random
  state); the unfinished generation's LLM calls and scores stay cached and are pushed. The CLI prints
  `PAUSED: <reason> ... Resume with: python evolve_<model>.py --resume <run_id>` and exits with code 2 (API) or 0
  (manual). `--resume` first pings every model of the ensemble and pauses again if one still fails.
- **Manual pause:** `touch population/<run_id>/PAUSE` → the run pauses before its next LLM call (between children,
  never inside an evaluation). `--resume` removes the file.

## 6. Results and versioning

- Non-evolution evaluations (baselines, manual versions, best-* scripts) write to `scores/<split>/scores_<id>.json`.
  **Milestone runs use a step-specific id** (e.g. `manual_instruct_s2b`), so every step keeps its scores file and its
  row in the generated `RESULTS.md` table; final versions use the deliverable name (e.g. `manual_instruct`).
  Re-running the same id overwrites; git keeps the history.
- `RESULTS.md` logs every significant run: what changed, the numbers, and the conclusion.
- Commit after each milestone; tag important ones with annotated tags (`v0.1-baseline`, `v0.2-manual-instruct`, …).
  Rollback is through git; no parallel versioned copies of files.
- Committed: code, `scores/`, `population/`, `candidates/`, `RESULTS.md`. Ignored: images, `logs/`, environments,
  keys, model weights, caches (see `.gitignore`).

## 7. Changing the contract

Adding a field is backward compatible. Renaming or removing one, or changing its meaning, requires bumping
`schema_version` (in `evaluate.py` for scores, in the controller for checkpoints) and updating this file.
