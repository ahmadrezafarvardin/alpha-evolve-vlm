"""Generate the RESULTS.md summary table from scores/<split>/scores_<id>.json (the source of truth).

    python scripts/results_table.py           # print the table
    python scripts/results_table.py --write   # replace the table between the markers in RESULTS.md

Offline replays are not scores files and are logged by hand in RESULTS.md, outside the markers.
"""
import re
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START, END = "<!-- results-table:start -->", "<!-- results-table:end -->"
SPLIT_ORDER = {"dev": 0, "evo": 1, "holdout": 2, "reserve": 3}


def gpu_name(env):
    gpu = (env or {}).get("gpu") or "?"
    return gpu.replace("NVIDIA GeForce ", "").replace("Tesla ", "")


def model_name(scores):
    """Model used by the scored program: from its source if still on disk, else guessed from the id."""
    program = Path(scores.get("program", ""))
    if program.is_file():
        source = program.read_text(errors="ignore")
        match = re.search(r"_model_name\s*=\s*['\"][^'\"]*Qwen3-VL-2B-(Instruct|Thinking)", source)
        if match:
            return match.group(1)
    return "Thinking" if "thinking" in scores["candidate_id"].lower() else "Instruct"


def rows():
    out = []
    for path in sorted(ROOT.glob("scores/*/scores_*.json")):
        d = json.loads(path.read_text())
        ok = d.get("status") == "ok"
        model = model_name(d)
        out.append({
            "date": d.get("timestamp", "")[:10], "id": d["candidate_id"], "model": model, "split": d["split"],
            "gpu": gpu_name(d.get("env")), "status": d.get("status"),
            "acc": f"**{d['accuracy']:.1%}** ({d['num_correct']}/{d['num_evaluated']})" if ok else d.get("status"),
            "time": f"{d['total_time']:.0f} s" if ok else "", "spq": f"{d['avg_time_per_query']:.2f}" if ok else "",
            "first": f"{d['first_query_time']:.1f} s" if ok and d.get("first_query_time") is not None else "",
            "errors": str(d.get("num_errors", "")),
        })
    out.sort(key=lambda r: (SPLIT_ORDER.get(r["split"], 9), r["date"], r["id"]))
    return out


def table():
    head = ("| Date | Id | Model | Split | GPU | Accuracy | Total time | s/query | First query | Errors |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n")
    return head + "".join(
        f"| {r['date']} | `{r['id']}` | {r['model']} | {r['split']} | {r['gpu']} | {r['acc']} | {r['time']} | "
        f"{r['spq']} | {r['first']} | {r['errors']} |\n" for r in rows())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="update RESULTS.md in place")
    args = parser.parse_args()
    text = table()
    if not args.write:
        print(text, end="")
        return
    path = ROOT / "RESULTS.md"
    doc = path.read_text()
    i, j = doc.index(START) + len(START), doc.index(END)
    path.write_text(doc[:i] + "\n" + text + doc[j:])
    print(f"updated {path.name}: {text.count(chr(10)) - 2} rows")


if __name__ == "__main__":
    main()
