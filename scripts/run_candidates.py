"""Evaluate every candidates/candidate_<id>.py and write scores/<split>/scores_<id>.json (see CONTRACT.md).

    python scripts/run_candidates.py                         # all unscored candidates, dev split (128 queries)
    python scripts/run_candidates.py --split evo             # evo split (400 queries)
    python scripts/run_candidates.py --only 0003 0007 --n 8  # selected ids, quick check

Each candidate runs in its own subprocess with a timeout, so crashes, CUDA OOMs and hangs are
isolated and recorded as status "error"/"timeout". A candidate whose scores file already exists
with the same code hash, split and query count is skipped, so the runner can resume after a disconnect.
"""
import sys
import json
import time
import hashlib
import argparse
import datetime
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluate import SCHEMA_VERSION, SPLITS, environment_info, load_charxiv_data  # noqa: E402

TAIL_CHARS = 4000


def candidate_id(path):
    return path.stem.removeprefix("candidate_")


def already_scored(scores_path, code_hash, split, num_samples):
    if not scores_path.exists():
        return False
    try:
        scores = json.loads(scores_path.read_text())
    except json.JSONDecodeError:
        return False
    return (scores.get("code_sha256") == code_hash and scores.get("split") == split
            and scores.get("num_samples") == num_samples)


def failure_scores(cid, path, code_hash, status, split, num_samples, wall_time, log):
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": cid,
        "status": status,
        "program": str(path),
        "code_sha256": code_hash,
        "split": split,
        "num_samples": num_samples,
        "accuracy": None,
        "wall_time": wall_time,
        "env": environment_info(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "error": {"log_tail": log[-TAIL_CHARS:]},
    }


def run_one(path, scores_dir, log_dir, split, num_samples, timeout, cid=None, scores_path=None):
    """Score one program in a fresh subprocess; `scores_path` overrides <scores_dir>/scores_<id>.json."""
    cid = cid or candidate_id(path)
    code_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    scores_path = Path(scores_path) if scores_path else scores_dir / f"scores_{cid}.json"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    if already_scored(scores_path, code_hash, split, num_samples):
        print(f"[{cid}] already scored, skipping")
        return json.loads(scores_path.read_text())

    cmd = [sys.executable, str(ROOT / "evaluate.py"), str(path), "--split", split, "--n", str(num_samples),
           "--out", str(scores_path), "--candidate-id", cid]
    print(f"[{cid}] running ...", flush=True)
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        log, status = proc.stdout + proc.stderr, "ok" if proc.returncode == 0 else "error"
    except subprocess.TimeoutExpired as e:
        log = (e.stdout or b"").decode(errors="replace") + (e.stderr or b"").decode(errors="replace")
        status = "timeout"
    wall_time = time.time() - start
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    (Path(log_dir) / f"{scores_path.stem.removeprefix('scores_')}_{split}.log").write_text(log)

    if status == "ok" and scores_path.exists():
        return json.loads(scores_path.read_text())
    if status == "ok":
        status = "error"  # exited cleanly but wrote no scores
    scores = failure_scores(cid, path, code_hash, status, split, num_samples, wall_time, log)
    scores_path.write_text(json.dumps(scores, indent=2))
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates-dir", default=str(ROOT / "candidates"))
    parser.add_argument("--scores-dir", default=str(ROOT / "scores"), help="scores go to <scores-dir>/<split>/")
    parser.add_argument("--log-dir", default=str(ROOT / "logs"))
    parser.add_argument("--split", default="dev", choices=SPLITS)
    parser.add_argument("--n", type=int, help="first N queries of the split (default: dev 128, others all)")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds per candidate")
    parser.add_argument("--only", nargs="*", help="candidate ids to run (default: all)")
    args = parser.parse_args()

    candidates = sorted(Path(args.candidates_dir).glob("candidate_*.py"))
    if args.only:
        candidates = [p for p in candidates if candidate_id(p) in args.only]
    if not candidates:
        sys.exit(f"No candidate_*.py files found in {args.candidates_dir}")
    scores_dir, log_dir = Path(args.scores_dir) / args.split, Path(args.log_dir)
    num_samples = len(load_charxiv_data(args.n if args.n is not None or args.split != "dev" else 128, args.split)[0])
    scores_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(candidates)} candidate(s), split {args.split}, {num_samples} queries each")
    results = [run_one(p, scores_dir, log_dir, args.split, num_samples, args.timeout) for p in candidates]

    print(f"\n{'id':<12}{'status':<9}{'accuracy':>9}{'time (s)':>10}{'errors':>8}")
    for s in results:
        acc = f"{s['accuracy']:.3f}" if s["accuracy"] is not None else "-"
        print(f"{s['candidate_id']:<12}{s['status']:<9}{acc:>9}{s.get('total_time', s['wall_time']):>10.1f}"
              f"{s.get('num_errors', '-'):>8}")


if __name__ == "__main__":
    main()
