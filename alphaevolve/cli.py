"""Command line shared by evolve_instruct.py and evolve_thinking.py."""
import argparse
import datetime
import json
import sys
from pathlib import Path

from .llm import LLM, APIUnavailable, CallFailed
from .loop import PAUSE_FILE, Evolution
from .prompt import PromptBuilder

MODEL_IDS = {"Instruct": "Qwen/Qwen3-VL-2B-Instruct", "Thinking": "Qwen/Qwen3-VL-2B-Thinking"}


def main(config, default_seed, prefix):
    parser = argparse.ArgumentParser(description=f"AlphaEvolve-style evolution of vlm_inference ({config['model']})")
    parser.add_argument("--resume", metavar="RUN_ID", help="continue population/<RUN_ID> from its checkpoint")
    parser.add_argument("--seed-program", default=default_seed, help="program to start from (new runs)")
    parser.add_argument("--generations", type=int, default=config["generations"], help="total generations")
    parser.add_argument("--children", type=int, help="children per generation (new runs; overrides the config)")
    parser.add_argument("--no-push", action="store_true", help="do not push checkpoints to evo/<run_id>")
    parser.add_argument("--export", choices=["accuracy", "speed", "overall"],
                        help=f"after the run, copy that best program to evolved_{prefix}.py")
    args = parser.parse_args()

    run_id = args.resume or f"{prefix}_{datetime.datetime.now():%Y%m%d-%H%M}"
    if args.resume:  # a resumed run keeps the configuration it started with, including command-line overrides
        saved = json.loads((Path(__file__).resolve().parent.parent / "population" / run_id / "config.json").read_text())
        config = {k: v for k, v in saved.items() if k not in ("run_id", "git_commit", "seed_program", "seed_sha256")}
    elif args.children:
        config = dict(config, children_per_generation=args.children)
    evo = Evolution(config, run_id, llm=LLM(), push=not args.no_push)
    evo.builder = PromptBuilder(MODEL_IDS[config["model"]], evo.rng)
    script = Path(sys.argv[0]).name
    if args.resume:
        pause_file = evo.path(PAUSE_FILE)
        if pause_file.exists():
            pause_file.unlink()
            print(f"removed {pause_file} (resuming is the explicit decision to continue)")
        # Re-test the API with one cheap call per model before touching the run; still failing -> pause again.
        for member in config["llm_ensemble"]:
            try:
                evo.llm.ping(member)
            except (APIUnavailable, CallFailed) as e:
                next_generation = json.loads(evo.path("checkpoint.json").read_text())["generation"] + 1
                evo.log_api_failure(next_generation, "ping", {"model": member["model"], "status": str(e)[:300]})
                paused(script, run_id, f"API check before resuming failed ({member['model']}): {e}", api=True)
        evo.resume()
    else:
        evo.start(args.seed_program)
    print(f"run {run_id}: generation {evo.generation}, target {args.generations}", flush=True)
    result = evo.run(args.generations)
    if result["status"] == "paused":
        paused(script, run_id, result["reason"], api=result["api_status"] is not None, generation=evo.generation)
    print(json.dumps({"best": evo.db.best(), "llm_usage": evo.usage()}, indent=1))
    if args.export:
        pid = evo.export(args.export, f"evolved_{prefix}.py")
        print(f"exported {pid} to evolved_{prefix}.py")


def paused(script, run_id, reason, api, generation=None):
    """Print the pause notice and exit: 2 for an API problem, 0 for a manual pause (PAUSE file)."""
    where = f" Last checkpoint: generation {generation}; work of the unfinished generation is cached." \
        if generation is not None else ""
    print(f"PAUSED: {reason}.{where}\nResume with: python {script} --resume {run_id}", flush=True)
    sys.exit(2 if api else 0)
