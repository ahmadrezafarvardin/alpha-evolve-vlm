"""Evolve vlm_inference for Qwen3-VL-2B-Thinking (AlphaEvolve-style; see CONTRACT.md §5 and alphaevolve/).

    python evolve_thinking.py                          # new run seeded from manual_thinking.py
    python evolve_thinking.py --resume thinking_20261015-0900

Thinking costs about 26 s/query on the RTX 3090 with a 1024-token reasoning budget (HF generate), so a full
400-query evo evaluation would take about 3 hours. The cascade therefore uses the first 4 / 32 / 128 evo queries,
and fitness is accuracy on evo[:128] (standard error about ±4 points). Speed is part of what evolution can improve
(reasoning budget, early stopping of second-guessing), which shortens every later evaluation.
"""
from alphaevolve.cli import main
from evolve_instruct import LLM_ENSEMBLE

CONFIG = {
    "model": "Thinking",
    "seed": 0,
    "generations": 6,
    "children_per_generation": 4,
    "num_inspirations": 2,
    "near_duplicate_similarity": 0.8,     # reject a child whose edit matches a sibling's this closely (before GPU)
    "llm_ensemble": LLM_ENSEMBLE,
    "llm_temperatures": [0.4, 0.7, 1.0],
    "cascade": {
        "stages": [{"name": "smoke", "n": 4, "timeout": 900},
                   {"name": "subset", "n": 32, "timeout": 3600},
                   {"name": "full", "n": 128, "timeout": 10800}],
        # margin 0 (Instruct: 0.03): a child must be at least as accurate as its parent on the 32-query subset, or
        # clearly faster at comparable accuracy. Neutral children then stop after ~16 min instead of a ~70 min full
        # stage, while speed-ups such as bounded reasoning still pass on the speed rule (user decision 2026-10-15).
        "promotion": {"margin": 0.0, "speed_margin": 0.06, "min_speedup": 0.10},
    },
    "database": {
        "num_islands": 2,
        "migration_interval": 2,
        "exploit_prob": 0.7,
        "exploit_top_k": 2,
        "speed_accuracy_tolerance": 0.02,
        "overall_time_weight": 0.05,
        "reference_time_per_query": 26.0,     # manual_thinking step 1 estimate (budget 1024), RTX 3090
        "features": [{"name": "time", "bins": [8, 15, 25, 40]},
                     {"name": "acc_quant", "bins": [0.3, 0.4, 0.5]}],
    },
}

if __name__ == "__main__":
    main(CONFIG, default_seed="manual_thinking.py", prefix="thinking")
