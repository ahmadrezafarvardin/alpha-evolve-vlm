# Overnight holdout queue for the FROZEN manual versions (tag v1.1-manual-frozen) and the Instruct baseline.
# Starts after the Thinking launcher and the Thinking evolution have ended (exact tmux names), never alongside an
# evolution run (timing would be corrupted). Each job commits and pushes its own scores file when it is written.
cd ~/evolving-vlm-inference
P=.venv/bin/python
F='Warning\|warn\|deprecated\|generation flags'
stamp() { date -u +%FT%TZ; }
while tmux has-session -t '=thinking_launcher' 2>/dev/null; do sleep 60; done
while tmux has-session -t '=full_thinking' 2>/dev/null || tmux has-session -t '=full_instruct' 2>/dev/null; do sleep 60; done
echo "$(stamp) evolution sessions ended; starting the holdout queue"

job() {  # job <program> <id> <extra evaluate args...>
  local program=$1 id=$2; shift 2
  local out=scores/holdout/scores_${id}.json
  echo "$(stamp) START $id"
  $P evaluate.py "$program" --split holdout "$@" --out "$out" --candidate-id "$id" 2>&1 | grep --line-buffered -v "$F"
  if [ -f "$out" ] && grep -q '"status": "ok"' "$out"; then
    git add "$out" && git commit -q -m "Holdout: $id (overnight queue, frozen manual versions)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && timeout 120 git push -q origin main \
      && echo "$(stamp) DONE $id (committed and pushed)" || echo "$(stamp) DONE $id (committed; push failed or timed out)"
  else
    echo "$(stamp) FAILED $id (no ok scores file); continuing with the next job"
  fi
}

job manual_thinking manual_thinking_h000 --n 400
job manual_thinking manual_thinking_h400 --offset 400 --n 400
job manual_instruct manual_instruct
job starting_scripts baseline_starting_scripts
echo "$(stamp) QUEUE_DONE"
