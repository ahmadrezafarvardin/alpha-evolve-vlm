"""Program database: islands, each with a MAP-Elites grid, plus a complete record of every program.

- **Record:** every proposed program is kept in `programs` (including rejected and failed ones), as in
  `checkpoint.json` (CONTRACT.md §5). Only programs that reached the full evo stage are *eligible*: they have a
  fitness and can be parents, inspirations, elites or `best`.
- **Islands** evolve separately (diversity); every `migration_interval` generations each island's best program
  is copied to the next island (ring).
- **MAP-Elites:** within an island, a program competes only with programs in the same feature cell
  (e.g. speed bin x quant-accuracy bin), so a fast but less accurate strategy is not wiped out by a slow, accurate one.
- **Selection:** with probability `exploit_prob` the parent is one of the island's top programs by fitness;
  otherwise a uniformly random elite of the island's grid (exploration). Inspirations are other elites of the
  island plus the global best.
"""
import bisect
import math

FULL_STAGE = "full"


TIME_NOISE = 0.10  # run-to-run timing noise is ~7 %; smaller time differences are not real


def fitness(scores):
    """Primary objective: accuracy, then speed (used for ranking candidates as parents)."""
    return (scores["accuracy"], -scores["avg_time_per_query"])


def clearly_faster(a, b):
    """True if scores `a` is faster than `b` by more than the timing noise."""
    return a["avg_time_per_query"] < (1 - TIME_NOISE) * b["avg_time_per_query"]


def beats(a, b):
    """Should `a` replace the incumbent `b`? Higher accuracy, or equal accuracy and clearly faster; on anything
    within noise the incumbent stays, so timing noise never decides."""
    return a["accuracy"] > b["accuracy"] or (a["accuracy"] == b["accuracy"] and clearly_faster(a, b))


def incumbent_max(ids, scores_of, better):
    """Best of `ids` in order, replacing the current best only when `better(new, current)`."""
    best = None
    for pid in ids:
        if best is None or better(scores_of(pid), scores_of(best)):
            best = pid
    return best


def feature_value(scores, name):
    if name == "time":
        return scores["avg_time_per_query"]
    if name.startswith("acc_"):  # per-category accuracy, e.g. acc_quant
        entry = scores.get("per_category", {}).get(name[4:])
        return entry["accuracy"] if entry else 0.0
    raise KeyError(f"unknown feature {name!r}")


class Database:
    def __init__(self, config, rng):
        self.config = config
        self.rng = rng
        self.programs = {}
        self.islands = [[] for _ in range(config["num_islands"])]
        self.map_elites = [{} for _ in range(config["num_islands"])]

    # --- bookkeeping -----------------------------------------------------------------------------
    def eligible(self, pid):
        return FULL_STAGE in self.programs[pid].get("scores", {})

    def full_scores(self, pid):
        return self.programs[pid]["scores"][FULL_STAGE]

    def cell(self, scores):
        key = []
        for feature in self.config["features"]:
            value = feature_value(scores, feature["name"])
            key.append(str(bisect.bisect_right(feature["bins"], value)))
        return ",".join(key)

    def add(self, pid, record, island):
        """Store a program record; if it reached the full stage, place it on its island's grid."""
        previous = self.programs.get(pid, {})
        record = dict(record, island=previous.get("island", island), islands=previous.get("islands", []))
        self.programs[pid] = record
        if not self.eligible(pid):
            return
        scores = self.full_scores(pid)
        record["features"] = {f["name"]: feature_value(scores, f["name"]) for f in self.config["features"]}
        self._place(pid, island)

    def _place(self, pid, island):
        if pid not in self.islands[island]:
            self.islands[island].append(pid)
        islands = self.programs[pid].setdefault("islands", [])  # absent in checkpoints written before this field
        if island not in islands:
            islands.append(island)  # `island` is the home island; this lists all, incl. migration
        cell = self.cell(self.full_scores(pid))
        current = self.map_elites[island].get(cell)
        if current is None or beats(self.full_scores(pid), self.full_scores(current)):
            self.map_elites[island][cell] = pid

    def migrate(self):
        """Ring migration: the best program of island i joins island i+1."""
        bests = [self.island_best(i) for i in range(len(self.islands))]
        for i, pid in enumerate(bests):
            if pid is not None:
                self._place(pid, (i + 1) % len(self.islands))

    # --- selection -------------------------------------------------------------------------------
    def island_best(self, island):
        members = [p for p in self.islands[island] if self.eligible(p)]
        return incumbent_max(members, self.full_scores, beats)

    def sample(self, island, num_inspirations):
        """(parent id, inspiration ids) for a child on `island`."""
        members = sorted(self.islands[island], key=lambda p: fitness(self.full_scores(p)), reverse=True)
        if not members:
            raise RuntimeError(f"island {island} has no evaluated programs (seed not evaluated?)")
        elites = sorted(set(self.map_elites[island].values()))
        if self.rng.random() < self.config["exploit_prob"]:
            parent = self.rng.choice(members[:self.config["exploit_top_k"]])
        else:
            parent = self.rng.choice(elites)
        pool = [p for p in elites + members[:3] if p != parent]
        pool = list(dict.fromkeys(pool))
        inspirations = self.rng.sample(pool, min(num_inspirations, len(pool)))
        best = self.best()["accuracy"]
        if best and best != parent and best not in inspirations:
            inspirations.append(best)
        return parent, inspirations

    def best(self):
        """Best eligible programs for accuracy, speed (among programs close to the best accuracy) and overall.
        Programs are considered in the order they were added; a later one wins only by more than the noise."""
        eligible = [p for p in self.programs if self.eligible(p)]
        if not eligible:
            return {"accuracy": None, "speed": None, "overall": None}
        acc = incumbent_max(eligible, self.full_scores, beats)
        floor = self.full_scores(acc)["accuracy"] - self.config["speed_accuracy_tolerance"]
        fast = [p for p in eligible if self.full_scores(p)["accuracy"] >= floor]
        speed = incumbent_max(fast, self.full_scores, clearly_faster)
        # The overall score moves by w*log2(1/0.9) when the time changes by the noise; require more than that.
        margin = self.config["overall_time_weight"] * math.log2(1 / (1 - TIME_NOISE))
        overall = incumbent_max(eligible, lambda p: p, lambda a, b: self.overall(a) > self.overall(b) + margin)
        return {"accuracy": acc, "speed": speed, "overall": overall}

    def overall(self, pid):
        """accuracy - w * log2(time / reference time): halving the time is worth `w` accuracy."""
        s = self.full_scores(pid)
        ref = self.config["reference_time_per_query"]
        return s["accuracy"] - self.config["overall_time_weight"] * math.log2(s["avg_time_per_query"] / ref)

    # --- checkpoint ------------------------------------------------------------------------------
    def to_dict(self):
        return {"programs": self.programs, "islands": self.islands,
                "map_elites": {str(i): cells for i, cells in enumerate(self.map_elites)}, "best": self.best()}

    @classmethod
    def from_dict(cls, data, config, rng):
        db = cls(config, rng)
        db.programs = data["programs"]
        db.islands = data["islands"]
        db.map_elites = [data["map_elites"][str(i)] for i in range(config["num_islands"])]
        return db
