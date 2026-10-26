"""Measure a program's post-processing offline by replaying it on recorded raw responses (no GPU).

    python scripts/replay_postprocess.py manual_instruct --scores scores/dev/scores_baseline_starting_scripts.json

With greedy decoding the model's raw output does not depend on post-processing, so applying
`program.postprocess(question, raw_response)` to the responses of a program that had *no* post-processing
gives exactly the accuracy a GPU run of "same generation + this post-processing" would get.
The scores file must therefore come from a program that returned raw model output (e.g. the baseline), or one
that recorded it in `raw_response` (programs exposing `last_raw_response`), with the same model and generation
settings as the program being measured.
"""
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluate import load_charxiv_data, load_program  # noqa: E402


def is_correct(response, gold):
    return response.strip().lower() == gold.lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("program", help="module name or .py path exposing postprocess(question, response)")
    parser.add_argument("--scores", required=True, help="scores JSON holding raw responses")
    parser.add_argument("--show", type=int, default=100, help="max changed answers to print per group")
    args = parser.parse_args()

    program, _ = load_program(args.program)
    scores = json.loads(Path(args.scores).read_text())
    queries, _ = load_charxiv_data(None, scores["split"])

    before = after = 0
    per_category = {}
    fixed, broken, changed_wrong = [], [], []
    for r in scores["responses"]:
        raw = r.get("raw_response", r["response"])
        new = program.postprocess(queries[r["key"]]["question"], raw)
        was, now = is_correct(raw, r["gold"]), is_correct(new, r["gold"])
        before += was
        after += now
        cat = per_category.setdefault(r["category"], [0, 0, 0])
        cat[0] += was
        cat[1] += now
        cat[2] += 1
        if new != raw:
            entry = (r["key"], r["qid"], r["gold"], raw, new)
            (fixed if now and not was else broken if was and not now else changed_wrong).append(entry)

    n = len(scores["responses"])
    print(f"{args.program} on {args.scores} ({scores['split']}, {n} queries)")
    print(f"accuracy: {before / n:.1%} -> {after / n:.1%}  ({before} -> {after}, {after - before:+d})")
    print(f"changed: {len(fixed) + len(broken) + len(changed_wrong)}  fixed: {len(fixed)}  "
          f"broken: {len(broken)}  changed but still wrong: {len(changed_wrong)}\n")
    for name, (b, a, total) in sorted(per_category.items()):
        print(f"  {name:<7} {b:>3} -> {a:<3} / {total}")

    for title, group in (("FIXED", fixed), ("BROKEN", broken), ("CHANGED, STILL WRONG", changed_wrong)):
        if group:
            print(f"\n=== {title} ({len(group)})")
        for key, qid, gold, old, new in group[:args.show]:
            print(f"[{key} q{qid}] gold={gold!r}  new={new!r}\n    was: ...{old[-120:]!r}")


if __name__ == "__main__":
    main()
