"""Evaluation cascade (CONTRACT.md §5): smoke (8 evo queries) -> subset (64) -> full (400).

Each stage runs `evaluate.py` in a fresh subprocess through `scripts/run_candidates.run_one`, so crashes,
OOMs and hangs are contained and every stage leaves one scores file:
`scores/evo/scores_<id>.n<N>.json` for partial stages and `scores/evo/scores_<id>.json` for the full stage.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_candidates import run_one  # noqa: E402


def summary(scores):
    """The part of a scores file kept in the checkpoint."""
    return {k: scores.get(k) for k in ("accuracy", "total_time", "avg_time_per_query", "per_category", "num_errors")}


def smoke_passes(scores):
    if scores["status"] != "ok":
        return False, f"status {scores['status']}"
    if scores.get("num_errors"):
        return False, f"{scores['num_errors']} queries raised"
    if len({r["response"] for r in scores["responses"]}) == 1:
        return False, "every response is identical"
    return True, "ok"


def promotes(child, parent, rule):
    """Promotion from the subset to the full stage (CONTRACT.md §5); `child`/`parent` are subset summaries."""
    if child["accuracy"] >= parent["accuracy"] - rule["margin"]:
        return True, "within margin of parent accuracy"
    faster = child["total_time"] <= (1 - rule["min_speedup"]) * parent["total_time"]
    if faster and child["accuracy"] >= parent["accuracy"] - rule["speed_margin"]:
        return True, "clearly faster at comparable accuracy"
    return False, (f"subset accuracy {child['accuracy']:.3f} vs parent {parent['accuracy']:.3f}, "
                   f"time {child['total_time']:.0f}s vs {parent['total_time']:.0f}s")


def scores_path(run_dir, cid, n, full):
    return Path(run_dir) / "scores" / "evo" / (f"scores_{cid}.json" if full else f"scores_{cid}.n{n}.json")


def run_cascade(program_path, cid, run_dir, config, parent_subset=None):
    """Evaluate a program stage by stage.

    Returns (stage_scores, stage_reached, status, reason): `stage_scores` maps stage name to its summary; a seed
    (`parent_subset=None`) always goes through every stage.
    """
    stages, rule = config["cascade"]["stages"], config["cascade"]["promotion"]
    log_dir = Path(run_dir) / "logs"
    stage_scores, reached = {}, None
    for stage in stages:
        full = stage["name"] == "full"
        scores = run_one(Path(program_path), None, log_dir, "evo", stage["n"], stage["timeout"], cid=cid,
                         scores_path=scores_path(run_dir, cid, stage["n"], full))
        if scores["status"] != "ok":
            return stage_scores, reached, scores["status"], f"{stage['name']}: {scores['status']}"
        stage_scores[stage["name"]] = summary(scores)
        reached = stage["name"]
        if stage["name"] == "smoke":
            ok, reason = smoke_passes(scores)
            if not ok:
                return stage_scores, reached, "rejected", f"smoke: {reason}"
        if stage["name"] == "subset" and parent_subset is not None:
            ok, reason = promotes(stage_scores["subset"], parent_subset, rule)
            if not ok:
                return stage_scores, reached, "stopped", f"subset: {reason}"
    return stage_scores, reached, "ok", "full evaluation"
