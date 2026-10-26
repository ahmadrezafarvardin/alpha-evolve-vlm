"""Tests for alphaevolve.database: placement, MAP-Elites competition, selection, migration, checkpoint round trip."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alphaevolve.database import Database  # noqa: E402

CONFIG = {
    "num_islands": 2, "exploit_prob": 0.7, "exploit_top_k": 2, "speed_accuracy_tolerance": 0.02,
    "overall_time_weight": 0.05, "reference_time_per_query": 0.5,
    "features": [{"name": "time", "bins": [0.3, 0.6, 1.2]}, {"name": "acc_quant", "bins": [0.25, 0.5]}],
}


def record(acc, t, quant=0.3, full=True):
    scores = {"accuracy": acc, "avg_time_per_query": t, "per_category": {"quant": {"accuracy": quant}}}
    return {"parent_id": None, "status": "ok", "scores": {"full": scores} if full else {"subset": scores}}


def make_db():
    return Database(CONFIG, random.Random(0))


def test_only_full_stage_programs_are_placed():
    db = make_db()
    db.add("a", record(0.5, 0.4), island=0)
    db.add("b", record(0.9, 0.4, full=False), island=0)
    assert db.islands[0] == ["a"] and "b" in db.programs and db.best()["accuracy"] == "a"


def test_elite_is_replaced_only_within_the_same_cell():
    db = make_db()
    db.add("slow", record(0.60, 2.0), island=0)
    db.add("fast", record(0.50, 0.4), island=0)    # other speed cell: both survive
    db.add("slow2", record(0.65, 2.0), island=0)   # same cell as "slow", better: replaces it
    assert sorted(db.map_elites[0].values()) == ["fast", "slow2"]


def test_best_pointers():
    db = make_db()
    db.add("acc", record(0.60, 2.0), island=0)
    db.add("near", record(0.59, 0.4), island=1)    # within tolerance of best accuracy and faster
    db.add("fast", record(0.30, 0.1), island=1)    # fastest but far less accurate
    best = db.best()
    assert best["accuracy"] == "acc" and best["speed"] == "near" and best["overall"] == "near"


def test_sample_returns_island_members_and_global_best():
    db = make_db()
    for i, acc in enumerate((0.4, 0.5, 0.45)):
        db.add(f"p{i}", record(acc, 0.2 + i), island=0)
    db.add("top", record(0.7, 0.4), island=1)
    for _ in range(20):
        parent, insp = db.sample(0, num_inspirations=2)
        assert parent in db.islands[0] and parent not in insp and "top" in insp


def test_migration_copies_island_best_to_next_island():
    db = make_db()
    db.add("a", record(0.5, 0.4), island=0)
    db.add("b", record(0.6, 0.4), island=1)
    db.migrate()
    assert "a" in db.islands[1] and "b" in db.islands[0]
    assert db.programs["a"]["island"] == 0 and db.programs["a"]["islands"] == [0, 1]


def test_checkpoint_round_trip_is_json_and_preserves_state():
    db = make_db()
    db.add("a", record(0.5, 0.4), island=0)
    db.add("b", record(0.6, 1.0), island=1)
    data = json.loads(json.dumps(db.to_dict()))
    restored = Database.from_dict(data, CONFIG, random.Random(0))
    assert restored.islands == db.islands and restored.map_elites == db.map_elites
    assert restored.best() == db.best()


def test_timing_noise_never_decides():
    db = make_db()
    db.add("old", record(0.55, 0.40), island=0)
    db.add("noisy", record(0.55, 0.38), island=0)     # same accuracy, 5 % faster: noise
    assert db.best() == {"accuracy": "old", "speed": "old", "overall": "old"}
    assert db.map_elites[0][db.cell(db.full_scores("old"))] == "old"
    db.add("fast", record(0.55, 0.30), island=0)      # same accuracy, 25 % faster: real
    assert db.best()["accuracy"] == "fast" and db.best()["speed"] == "fast"
    db.add("better", record(0.56, 0.60), island=0)    # more accurate wins regardless of time
    assert db.best()["accuracy"] == "better"


def test_checkpoint_without_islands_field_still_migrates():
    db = make_db()
    db.add("a", record(0.5, 0.4), island=0)
    data = json.loads(json.dumps(db.to_dict()))
    del data["programs"]["a"]["islands"]                  # as written before records listed their islands
    restored = Database.from_dict(data, CONFIG, random.Random(0))
    restored.migrate()
    assert restored.programs["a"]["islands"] == [1]
