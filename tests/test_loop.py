"""End-to-end test of alphaevolve.loop with a fake LLM and a fake cascade (no GPU, no API key)."""
import json
import random
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import alphaevolve.loop as loop  # noqa: E402

SEED = '''# EVOLVE-BLOCK-START
_model_name = "Qwen/Qwen3-VL-2B-Instruct"
TOKENS = 64


def vlm_inference(image_path, question):
    return str(TOKENS)
# EVOLVE-BLOCK-END
'''

CONFIG = {
    "model": "Instruct", "seed": 1, "children_per_generation": 4, "num_inspirations": 1,
    "llm_temperatures": [0.5, 0.9], "llm_ensemble": [{"model": "fast", "weight": 0.8}, {"model": "strong", "weight": 0.2}],
    "cascade": {"stages": [{"name": "smoke", "n": 8, "timeout": 1}, {"name": "subset", "n": 64, "timeout": 1},
                           {"name": "full", "n": 400, "timeout": 1}],
                "promotion": {"margin": 0.03, "speed_margin": 0.06, "min_speedup": 0.1}},
    "database": {"num_islands": 2, "migration_interval": 2, "exploit_prob": 0.7, "exploit_top_k": 2,
                 "speed_accuracy_tolerance": 0.02, "overall_time_weight": 0.05, "reference_time_per_query": 0.5,
                 "features": [{"name": "time", "bins": [0.3, 0.6]}]},
}


class FakeLLM:
    """Proposes TOKENS = <random value>; occasionally an unusable answer."""
    def __init__(self):
        self.rng, self.calls = random.Random(7), 0  # its own generator, like a real LLM's independent randomness

    def generate(self, system, prompt, member, temperature, on_failure=None):
        self.calls += 1
        current = re.search(r"TOKENS = (\d+)", prompt).group(1)
        if self.calls % 5 == 0:
            return "no edit, sorry", {"model": "fake", "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0}
        new = self.rng.choice([16, 32, 48, 96, 128])
        text = f"<<<<<<< SEARCH\nTOKENS = {current}\n=======\nTOKENS = {new}\n>>>>>>> REPLACE"
        return text, {"model": "fake", "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.001}


class FakeBuilder:
    def system(self):
        return "system"

    def build(self, parent_code, parent_scores, parent_scores_file, inspirations, tried=()):
        self.last_tried = list(tried)
        return parent_code, {"directions": []}


def fake_cascade(program_path, cid, run_dir, config, parent_subset=None):
    """Accuracy rises with TOKENS (up to 96) and time with TOKENS: a real accuracy/speed trade-off."""
    tokens = int(re.search(r"TOKENS = (\d+)", Path(program_path).read_text()).group(1))
    acc, t = min(tokens, 96) / 160, tokens / 100
    summary = {"accuracy": acc, "total_time": t * 64, "avg_time_per_query": t, "per_category": {}, "num_errors": 0}
    if parent_subset and acc < parent_subset["accuracy"] - 0.03:
        return {"smoke": summary, "subset": summary}, "subset", "stopped", "worse"
    path = Path(run_dir) / "scores" / "evo" / f"scores_{cid}.json"
    path.write_text(json.dumps({"responses": []}))
    return {"smoke": summary, "subset": summary, "full": summary}, "full", "ok", "full evaluation"


def make(tmp_path, monkeypatch, run_id="instruct_test"):
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    evo = loop.Evolution(CONFIG, run_id, prompt_builder=FakeBuilder(), cascade=fake_cascade, push=False)
    evo.llm = FakeLLM()
    return evo


def test_run_records_every_child(tmp_path, monkeypatch):
    seed = tmp_path / "seed.py"
    seed.write_text(SEED)
    evo = make(tmp_path, monkeypatch)
    evo.start(seed)
    evo.run(generations=3)
    ckpt = json.loads((evo.dir / "checkpoint.json").read_text())
    assert ckpt["generation"] == 3 and len(ckpt["programs"]) == 1 + 3 * 4
    assert {"invalid_diff", "ok"} <= {p["status"] for p in ckpt["programs"].values()}
    assert ckpt["llm_usage"]["calls"] == 12
    assert len((evo.dir / "events.jsonl").read_text().splitlines()) == 1 + 12 + 1  # seed, children, 1 migration


def test_resume_picks_same_parents_and_reuses_llm_calls(tmp_path, monkeypatch):
    seed = tmp_path / "seed.py"
    seed.write_text(SEED)
    evo = make(tmp_path, monkeypatch)
    evo.start(seed)
    evo.run(generations=2)
    snapshot = tmp_path / "snapshot"
    shutil.copytree(evo.dir, snapshot)
    evo.run(generations=3)
    original = {k: (v["parent_id"], v["status"]) for k, v in evo.db.programs.items() if v.get("generation") == 3}

    restored = tmp_path / "population" / "instruct_test"
    llm_records = {f.name: f.read_text() for f in (restored / "llm").glob("g003_*.json")}
    shutil.rmtree(restored)
    shutil.copytree(snapshot, restored)
    for name, text in llm_records.items():            # these LLM calls finished before the crash
        (restored / "llm" / name).write_text(text)
    resumed = make(tmp_path, monkeypatch)
    resumed.resume()
    resumed.run(generations=3)
    redone = {k: (v["parent_id"], v["status"]) for k, v in resumed.db.programs.items() if v.get("generation") == 3}
    assert redone == original
    assert resumed.llm.calls == 0
    # The generation-3 calls were made before the "crash" and after the last checkpoint: they must still be counted.
    ckpt = json.loads((restored / "checkpoint.json").read_text())
    assert ckpt["llm_usage"]["calls"] == 12 and ckpt["llm_usage"]["by_model"]


def test_best_pointers_after_run(tmp_path, monkeypatch):
    seed = tmp_path / "seed.py"
    seed.write_text(SEED)
    evo = make(tmp_path, monkeypatch)
    evo.start(seed)
    evo.run(generations=4)
    best = evo.db.best()
    acc = evo.db.full_scores(best["accuracy"])["accuracy"]
    assert acc >= 64 / 160  # never worse than the seed
    assert evo.db.full_scores(best["speed"])["accuracy"] >= acc - 0.02


def test_prompt_lists_ideas_already_tried_from_the_parent(tmp_path, monkeypatch):
    seed = tmp_path / "seed.py"
    seed.write_text(SEED)
    evo = make(tmp_path, monkeypatch)
    evo.start(seed)
    evo.run(generations=2)
    tried = evo.tried_from("seed")
    assert tried and all(isinstance(idea, str) and outcome for idea, outcome in tried)
    assert any(o in ("a repeat", "same accuracy as the current program") or "accuracy" in o or "stopped" in o
               or "could not" in o for _, o in tried)
