"""Evolve vlm_inference for Qwen3-VL-2B-Instruct (AlphaEvolve-style; see CONTRACT.md §5 and alphaevolve/).

    python evolve_instruct.py                          # new run seeded from manual_instruct.py
    python evolve_instruct.py --resume instruct_20261015-0900
    python evolve_instruct.py --resume <run_id> --generations 20 --export overall

Needs ARVAN_API_KEY (in .env) and a GPU. Fitness is measured on the evo split only.
"""
from alphaevolve.cli import main

# Mutation LLM through ArvanCloud's OpenAI-compatible gateway (alphaevolve/llm.py). Names as listed by
# `scripts/check_llm.py`; prices in USD per million tokens (thinking tokens billed as output).
LLM_ENSEMBLE = [
    # max_tokens caps all output, thinking included. A test mutation call used 3.2k input, 198 output, 0 thinking tokens.
    # The model is drawn per call; 15 % go to the stronger Flash model for diversity (the pilot repeated one idea).
    {"model": "Gemini-3.1-Flash-Lite-Preview", "weight": 0.85, "price_in": 0.25, "price_out": 1.50, "max_tokens": 4096},
    # Flash Preview thinks by default: in the full run it spent 3,929 of 4,096 tokens reasoning and its edit was cut
    # off. reasoning_effort "low" gives a complete edit (468 tokens, $0.003 on the same prompt).
    {"model": "Gemini-3-Flash-Preview", "weight": 0.15, "price_in": 0.50, "price_out": 3.00, "max_tokens": 8192,
     "reasoning_effort": "low"},
]

CONFIG = {
    "model": "Instruct",
    "seed": 0,
    "generations": 12,
    "children_per_generation": 6,
    "num_inspirations": 2,
    "near_duplicate_similarity": 0.8,     # reject a child whose edit matches a sibling's this closely (before GPU)
    "llm_ensemble": LLM_ENSEMBLE,
    "llm_temperatures": [0.4, 0.7, 1.0],
    "cascade": {
        # About 20 s, 45 s and 3 min for a step-2c-like program on the RTX 3090.
        "stages": [{"name": "smoke", "n": 8, "timeout": 600},
                   {"name": "subset", "n": 64, "timeout": 1200},
                   {"name": "full", "n": 400, "timeout": 3600}],
        "promotion": {"margin": 0.03, "speed_margin": 0.06, "min_speedup": 0.10},
    },
    "database": {
        "num_islands": 3,
        "migration_interval": 3,
        "exploit_prob": 0.7,
        "exploit_top_k": 3,
        "speed_accuracy_tolerance": 0.02,
        "overall_time_weight": 0.05,          # halving the time is worth 5 accuracy points
        "reference_time_per_query": 0.44,     # manual_instruct step 2c on reserve[:400], RTX 3090
        "features": [{"name": "time", "bins": [0.3, 0.45, 0.7, 1.2]},
                     {"name": "acc_quant", "bins": [0.3, 0.4, 0.5]}],
    },
}

if __name__ == "__main__":
    main(CONFIG, default_seed="manual_instruct.py", prefix="instruct")
