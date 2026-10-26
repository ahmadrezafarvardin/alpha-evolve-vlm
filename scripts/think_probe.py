"""Measure manual_thinking's reasoning lengths with a large budget (used to choose _THINK_BUDGET).

    python scripts/think_probe.py 32 3072 analysis/think_probe_dev32.json   # first 32 dev queries, budget 3072

Writes evaluate()'s detailed results plus, per response, `think_tokens` (tokens before </think>) and `forced`
(whether the budget ran out and the answer was forced). Measurement only; not a scores file (CONTRACT.md §4).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import evaluate  # noqa: E402
import manual_thinking  # noqa: E402


def main():
    n, budget, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    manual_thinking._THINK_BUDGET = budget
    res = evaluate.evaluate(manual_thinking, num_samples=n, details=True)
    tokenizer = manual_thinking._get_model()[1].tokenizer
    for r in res["responses"]:
        raw = r.get("raw_response", "")
        r["think_tokens"] = len(tokenizer(raw.split("</think>")[0]).input_ids)
        r["forced"] = "I have to give the final answer now." in raw
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("accuracy", "num_correct", "total_time", "avg_time_per_query")}))


if __name__ == "__main__":
    main()
