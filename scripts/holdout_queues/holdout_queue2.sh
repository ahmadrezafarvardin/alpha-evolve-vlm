# Second holdout queue: evolved_instruct (tag v1.2-evolved-instruct) on all 800 holdout queries, after the first
# queue (exact tmux name) has ended. Commits and pushes only its own scores file.
cd ~/evolving-vlm-inference
stamp() { date -u +%FT%TZ; }
while tmux has-session -t '=holdout_queue' 2>/dev/null || tmux has-session -t '=full_thinking' 2>/dev/null \
      || tmux has-session -t '=thinking_launcher' 2>/dev/null; do sleep 60; done
out=scores/holdout/scores_evolved_instruct.json
echo "$(stamp) START evolved_instruct"
.venv/bin/python evaluate.py evolved_instruct --split holdout --out "$out" --candidate-id evolved_instruct 2>&1 \
  | grep --line-buffered -v 'Warning\|warn\|deprecated\|generation flags'
if [ -f "$out" ] && grep -q '"status": "ok"' "$out"; then
  git add "$out" && git commit -q -m "Holdout: evolved_instruct (second overnight queue)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && timeout 120 git push -q origin main \
    && echo "$(stamp) DONE evolved_instruct (committed and pushed)" || echo "$(stamp) DONE evolved_instruct (push failed)"
else
  echo "$(stamp) FAILED evolved_instruct"
fi
echo "$(stamp) QUEUE2_DONE"
