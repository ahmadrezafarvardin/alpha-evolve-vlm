#!/usr/bin/env bash
# Reproduce the results of this submission (RTX 3090 reference; see README.md, RESULTS.md, docs/REPORT_NOTES.md).
#
#   ./reproduce.sh                # setup check + evaluate the 7 submitted inference programs and the naive baseline
#                                 # on the official dev set (128 queries)
#   ./reproduce.sh --holdout      # also evaluate them on the 800-query holdout split (the report's primary evidence)
#   ./reproduce.sh --evolve       # also re-run both evolution systems (needs ARVAN_API_KEY in .env; hours of GPU)
#
# Greedy decoding throughout, so accuracy is deterministic on the same GPU type; times vary by ~7 %.
# Expected durations on an RTX 3090: Instruct-based programs ~1 min per 128 queries; Thinking-based programs
# ~20-30 s per query (evolved_thinking ~19 s/q, manual_thinking ~29 s/q on holdout), so the dev step takes ~2-3 h.
set -euo pipefail
cd "$(dirname "$0")"

usage() {
  cat <<'USAGE'
Usage: bash reproduce.sh [--holdout] [--evolve] [--help]

  (default)   environment/data check, GPU-free tests, then the 7 submitted inference programs and the naive baseline
              on the official dev set (the 128 given samples)                       ~2-3 h on an RTX 3090
  --holdout   additionally the same programs on the 800-query holdout split, i.e. the report's main results table
                                                                                    ~14 h more (Thinking programs)
  --evolve    additionally re-run both evolution systems (needs ARVAN_API_KEY in .env)          ~5 h more
  --help      this message

Instruct-based programs take ~0.4 s/query, Thinking-based ones 19-29 s/query. Results: reproduce_out/*.json.
USAGE
}

HOLDOUT=0; EVOLVE=0
for arg in "$@"; do
  case "$arg" in
    --holdout) HOLDOUT=1 ;;
    --evolve) EVOLVE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done
usage

PY=${PYTHON:-python}
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; fi

echo "== 1. Environment and data"
"$PY" -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"
if [ "$(ls charxiv/images | grep -c '\.jpg$')" -lt 2323 ]; then
  "$PY" scripts/download_images.py          # CharXiv images (2,323), verified against the metadata
fi
"$PY" -m pytest -q tests/                    # GPU-free tests: extraction, evolution machinery, API failure handling

PROGRAMS="manual_instruct manual_thinking evolved_instruct evolved_thinking best_accuracy best_speed best_overall"

echo "== 2. Official dev set (128 queries): python evaluate.py <program>"
mkdir -p reproduce_out
"$PY" evaluate.py starting_scripts --out reproduce_out/dev_starting_scripts.json
for p in $PROGRAMS; do
  "$PY" evaluate.py "$p" --out "reproduce_out/dev_$p.json"
done

if [ "$HOLDOUT" = 1 ]; then
  echo "== 3. Holdout (800 queries; never used for design decisions)"
  "$PY" evaluate.py starting_scripts --split holdout --out reproduce_out/holdout_starting_scripts.json
  for p in $PROGRAMS; do
    "$PY" evaluate.py "$p" --split holdout --out "reproduce_out/holdout_$p.json"
  done
fi

if [ "$EVOLVE" = 1 ]; then
  echo "== 4. Evolution systems (new run ids; results are stochastic only through the mutation LLM)"
  "$PY" scripts/check_llm.py                 # key present, models available
  "$PY" evolve_instruct.py --no-push          # 12 generations x 6 children (~3 h)
  "$PY" evolve_thinking.py --no-push --generations 2 --children 3   # as run for the report (~2 h)
fi

echo "== Summary (accuracy, s/query) of the files written to reproduce_out/"
"$PY" - <<'EOF'
import json, glob
for f in sorted(glob.glob("reproduce_out/*.json")):
    d = json.load(open(f))
    print(f"{f:<45} {d['accuracy']:.1%}  {d['avg_time_per_query']:.2f} s/query")
EOF
