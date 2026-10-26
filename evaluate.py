"""Evaluator for VLM inference using CharXiv benchmark.

Usage:
    python evaluate.py <module>                      # e.g. starting_scripts, manual_instruct (dev split)
    python evaluate.py manual_instruct --split evo --out scores/evo/scores_manual_instruct.json
    python evaluate.py starting_scripts --n 8        # quick smoke run on the first 8 queries

Splits are disjoint by figure (all from descriptive_val.json, 4 questions per figure):
    dev      first 32 figures   (128 queries; the official evaluate() default)
    evo      100 figures        (400 queries; fitness for evolution and manual tuning)
    holdout  200 figures        (800 queries; final generalization check only, never tuned on)
    reserve  remaining 668      (unused)
"""
import sys
import json
import time
import hashlib
import random
import argparse
import datetime
import platform
import importlib
import importlib.util
import traceback
from pathlib import Path

CHARXIV_PATH = Path(__file__).parent / "charxiv"
sys.path.insert(0, str(CHARXIV_PATH / "src"))

from descriptive_utils import build_descriptive_quries

SCHEMA_VERSION = 1

DEV_FIGURES = 32
SPLIT_SEED = 0
SPLIT_SIZES = {"evo": 100, "holdout": 200}  # in figures, drawn from the non-dev figures
SPLITS = ("dev", "evo", "holdout", "reserve")
# sha256 of the split assignment; guards against a Python version shuffling differently.
SPLIT_FINGERPRINT = "08613593bce3f049"

# Question template id -> category, mirroring charxiv/src/descriptive_utils.py:get_rubric
QID_CATEGORY = {1: "title", 11: "bool", 13: "enum", 16: "trend", 18: "layout"}
QID_CATEGORY.update({q: "ocr" for q in (2, 3, 4, 5, 6, 7)})
QID_CATEGORY.update({q: "quant" for q in (8, 9, 10, 12, 14, 15, 17, 19)})


def split_figure_ids(data, split):
    """Figure ids (keys of descriptive_val.json) in a split; deterministic and disjoint by figure."""
    figure_ids = list(data)
    if split == "dev":
        return figure_ids[:DEV_FIGURES]
    rest = figure_ids[DEV_FIGURES:]
    random.Random(SPLIT_SEED).shuffle(rest)
    n_evo, n_holdout = SPLIT_SIZES["evo"], SPLIT_SIZES["holdout"]
    bounds = {"evo": (0, n_evo), "holdout": (n_evo, n_evo + n_holdout), "reserve": (n_evo + n_holdout, len(rest))}
    start, end = bounds[split]
    return sorted(rest[start:end], key=int)


def check_split_fingerprint(data):
    assignment = {s: split_figure_ids(data, s) for s in SPLITS}
    fingerprint = hashlib.sha256(json.dumps(assignment).encode()).hexdigest()[:16]
    if fingerprint != SPLIT_FINGERPRINT:
        raise RuntimeError(f"Split assignment changed (fingerprint {fingerprint} != {SPLIT_FINGERPRINT}); "
                           "results would not be comparable across machines.")


def load_charxiv_data(num_samples=128, split="dev", offset=0):
    with open(CHARXIV_PATH / "data" / "descriptive_val.json") as f:
        data = json.load(f)
    if split != "dev":
        check_split_fingerprint(data)
        data = {fid: data[fid] for fid in split_figure_ids(data, split)}
    queries = build_descriptive_quries(data, str(CHARXIV_PATH / "images"))
    if offset or num_samples is not None:
        end = None if num_samples is None else offset + num_samples
        queries = dict(list(queries.items())[offset:end])
    return queries, data


def evaluate(program, num_samples=128, details=False, split="dev", offset=0):
    queries, ground_truth_data = load_charxiv_data(num_samples, split, offset)

    num_errors = 0
    start_time = time.time()
    for i, (query_key, query) in enumerate(queries.items(), 1):
        query_start = time.time()
        if hasattr(program, "last_raw_response"):
            program.last_raw_response = None
        try:
            response = program.vlm_inference(
                image_path=query["figure_path"],
                question=query["question"],
            )
            query["response"] = response
        except Exception as e:
            print(f"Error on {query_key}: {e}")
            traceback.print_exc()
            query["response"] = "ERROR"
            num_errors += 1
        query["latency"] = time.time() - query_start
        # Optional: programs that post-process expose the model's raw output, so extraction can be replayed offline.
        raw = getattr(program, "last_raw_response", None)
        if raw is not None:
            query["raw_response"] = raw
        if i % 16 == 0 or i == len(queries):
            print(f"[{i}/{len(queries)}] {time.time() - start_time:.0f}s, {num_errors} errors", file=sys.stderr, flush=True)
    total_time = time.time() - start_time

    correct = 0
    total = 0
    for query_key, query in queries.items():
        if "response" not in query:
            continue
        figure_id, subq_idx = query_key.split("_")
        gt_entry = ground_truth_data.get(figure_id)
        if gt_entry is None:
            continue
        gt_answer = gt_entry["answers"][int(subq_idx)]
        model_response = query["response"].strip()
        query["gold"] = gt_answer
        query["correct"] = model_response.lower() == gt_answer.lower()
        if query["correct"]:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    metrics = {"accuracy": accuracy, "num_evaluated": total, "num_errors": num_errors, "total_time": total_time, "avg_time_per_query": total_time / total if total > 0 else 0.0}
    if details:
        metrics.update(_details(queries))
    return metrics


def _details(queries):
    """Per-query records plus per-qid and per-category accuracy breakdowns."""
    responses, per_qid, per_category = [], {}, {}
    for query_key, query in queries.items():
        if "correct" not in query:
            continue
        qid = query["qid"]
        responses.append({
            "key": query_key, "qid": qid, "category": QID_CATEGORY[qid],
            "response": query["response"], "gold": query["gold"],
            "correct": query["correct"], "latency": query["latency"],
            **({"raw_response": query["raw_response"]} if "raw_response" in query else {}),
        })
        for bucket, name in ((per_qid, str(qid)), (per_category, QID_CATEGORY[qid])):
            entry = bucket.setdefault(name, {"correct": 0, "total": 0})
            entry["correct"] += int(query["correct"])
            entry["total"] += 1
    for bucket in (per_qid, per_category):
        for entry in bucket.values():
            entry["accuracy"] = entry["correct"] / entry["total"]
    per_qid = dict(sorted(per_qid.items(), key=lambda kv: int(kv[0])))
    return {
        "num_correct": sum(r["correct"] for r in responses),
        # The first call includes lazy model loading in most programs.
        "first_query_time": responses[0]["latency"] if responses else None,
        "per_qid": per_qid,
        "per_category": dict(sorted(per_category.items())),
        "responses": responses,
    }


def load_program(target):
    """Import a program by module name (`starting_scripts`) or file path (`candidates/x.py`)."""
    if target.endswith(".py") or "/" in target:
        path = Path(target).resolve()
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, path
    module = importlib.import_module(target)
    return module, Path(module.__file__).resolve()


def environment_info():
    info = {"python": platform.python_version(), "hostname": platform.node()}
    try:
        import torch
        import transformers
        info["torch"] = torch.__version__
        info["transformers"] = transformers.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("program", help="module name (e.g. starting_scripts) or path to a .py file")
    parser.add_argument("--split", default="dev", choices=SPLITS)
    parser.add_argument("--n", type=int, help="first N queries of the split (default: dev 128, others all)")
    parser.add_argument("--offset", type=int, default=0, help="skip the first OFFSET queries (e.g. reserve 400-800: "
                        "--offset 400 --n 400)")
    parser.add_argument("--out", help="write the full scores JSON (see CONTRACT.md) to this path")
    parser.add_argument("--candidate-id", help="id stored in the scores JSON (default: file stem)")
    args = parser.parse_args()

    num_samples = args.n if args.n is not None else (128 if args.split == "dev" else None)
    wall_start = time.time()
    module, path = load_program(args.program)
    result = evaluate(module, num_samples=num_samples, details=args.out is not None, split=args.split,
                      offset=args.offset)
    summary = {k: result[k] for k in ("accuracy", "num_evaluated", "num_errors", "total_time", "avg_time_per_query")}
    print(json.dumps(summary, indent=2))

    if args.out:
        scores = {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": args.candidate_id or path.stem,
            "status": "ok",
            "program": str(path),
            "code_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "split": args.split,
            "num_samples": result["num_evaluated"],
            "offset": args.offset,
            **result,
            "wall_time": time.time() - wall_start,
            "env": environment_info(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "error": None,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(scores, indent=2))
        tmp.replace(out)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
