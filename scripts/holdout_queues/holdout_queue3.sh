# Third holdout queue: evolved_thinking (= g001_c00 of thinking_20261015-1453) on all 800 holdout queries, run from the
# candidate file (same code hash as the later evolved_thinking.py export), after holdout_queue2 (exact name) ended.
cd ~/evolving-vlm-inference
stamp() { date -u +%FT%TZ; }
while tmux has-session -t '=holdout_queue2' 2>/dev/null || tmux has-session -t '=holdout_queue' 2>/dev/null; do sleep 60; done
src=population/thinking_20261015-1453/candidates/candidate_g001_c00.py
out=scores/holdout/scores_evolved_thinking.json
echo "$(stamp) START evolved_thinking ($src)"
.venv/bin/python evaluate.py "$src" --split holdout --out "$out" --candidate-id evolved_thinking 2>&1 \
  | grep --line-buffered -v 'Warning\|warn\|deprecated\|generation flags'
if [ -f "$out" ] && grep -q '"status": "ok"' "$out"; then
  git add "$out" && git commit -q -m "Holdout: evolved_thinking (g001_c00 of thinking_20261015-1453; third overnight queue)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && timeout 120 git push -q origin main \
    && echo "$(stamp) DONE evolved_thinking (committed and pushed)" || echo "$(stamp) DONE evolved_thinking (push failed)"
else
  echo "$(stamp) FAILED evolved_thinking"
fi
echo "$(stamp) QUEUE3_DONE"
