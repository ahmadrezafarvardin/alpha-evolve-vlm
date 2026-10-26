# Selection runs on reserve queries 800-1000 (untouched until now) for best_accuracy / best_speed (user decision
# 2026-10-16). Writes scores only; committed afterwards. Run from the repo root.
cd ~/evolving-vlm-inference
F='Warning\|warn\|deprecated\|generation flags'
stamp() { date -u +%FT%TZ; }
run() {  # run <program> <id>
  echo "$(stamp) START $2"
  .venv/bin/python evaluate.py "$1" --split reserve --offset 800 --n 200 --out "scores/reserve/scores_$2.json" \
    --candidate-id "$2" 2>&1 | grep --line-buffered -v "$F"
  echo "$(stamp) END $2"
}
run manual_instruct manual_instruct_r800
run evolved_instruct evolved_instruct_r800
run population/thinking_20261015-1453/candidates/candidate_g001_c00.py evolved_thinking_r800
run manual_thinking manual_thinking_r800
echo "$(stamp) SELECTION_DONE"
