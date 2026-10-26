"""The evolution loop (CONTRACT.md §5): sample -> prompt -> LLM -> diff -> checks -> cascade -> database.

One run lives in `population/<run_id>/`. After every generation the checkpoint is written atomically and the run
folder is committed to the branch `evo/<run_id>` and pushed, without touching the main working tree or index.
`--resume <run_id>` redoes the first unfinished generation: the restored random state makes it pick the same
parents and prompts, recorded LLM responses (`llm/<id>.json`) are reused, and finished stages are not re-run.
"""
import datetime
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from .checks import check_program
from .database import Database
from .diff import DiffError, apply_response, change_set, change_similarity
from .evaluator import run_cascade, scores_path
from .llm import APIUnavailable, CallFailed

ROOT = Path(__file__).resolve().parent.parent
PAUSE_FILE = "PAUSE"  # `touch population/<run_id>/PAUSE` pauses the run cleanly before its next LLM call


class Paused(Exception):
    """Stop the run cleanly: the last checkpoint stays the resume point, finished work of the generation is cached."""
    def __init__(self, reason, status=None):
        super().__init__(reason)
        self.reason, self.status = reason, status


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_json_atomic(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def rng_state_to_json(state):
    version, internal, gauss = state
    return [version, list(internal), gauss]


def rng_state_from_json(data):
    return (data[0], tuple(data[1]), data[2])


class Evolution:
    def __init__(self, config, run_id, llm=None, prompt_builder=None, cascade=run_cascade, push=True):
        self.config, self.run_id = config, run_id
        self.dir = ROOT / "population" / run_id
        self.llm, self.builder, self.cascade, self.push_enabled = llm, prompt_builder, cascade, push
        self.rng = random.Random(config["seed"])
        self.db = Database(config["database"], self.rng)
        self.generation = -1

    # --- files ------------------------------------------------------------------------------------
    def path(self, *parts):
        return self.dir.joinpath(*parts)

    def event(self, kind, **fields):
        with open(self.path("events.jsonl"), "a") as f:
            f.write(json.dumps({"time": now(), "event": kind, **fields}) + "\n")

    def usage(self):
        """LLM usage summed over the `llm/` records: they are written right after each call, so the total stays exact
        even when a crash happens between a call and the next checkpoint."""
        total = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "by_model": {}}
        for path in sorted(self.path("llm").glob("*.json")):
            call = json.loads(path.read_text())
            total["calls"] += 1
            for key in ("input_tokens", "output_tokens", "cost_usd"):
                total[key] += call.get(key, 0)
            total["by_model"][call.get("model", "?")] = total["by_model"].get(call.get("model", "?"), 0) + 1
        return total

    def checkpoint(self):
        write_json_atomic(self.path("checkpoint.json"), {
            "schema_version": 1, "run_id": self.run_id, "generation": self.generation, **self.db.to_dict(),
            "rng_state": rng_state_to_json(self.rng.getstate()), "llm_usage": self.usage(), "timestamp": now(),
        })

    def push(self):
        """Commit population/<run_id>/ to branch evo/<run_id> (temporary index; main is untouched) and push it."""
        if not self.push_enabled:
            return
        branch = f"evo/{self.run_id}"
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tempfile.gettempdir()) / f"evo-index-{self.run_id}"))
        git = lambda *args: subprocess.run(["git", *args], cwd=ROOT, env=env, check=True,  # noqa: E731
                                           capture_output=True, text=True).stdout.strip()
        parent = subprocess.run(["git", "rev-parse", "--verify", "-q", branch], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip() or git("rev-parse", "HEAD")
        git("read-tree", parent)
        git("add", "-f", str(self.dir.relative_to(ROOT)))
        tree = git("write-tree")
        commit = git("commit-tree", tree, "-p", parent, "-m", f"{self.run_id}: generation {self.generation}")
        git("update-ref", f"refs/heads/{branch}", commit)
        # A failed or hung push must not stop the run: the local checkpoint is intact and the next push carries it.
        # (A push once hung for 10+ minutes on a dead connection; without a timeout the run would freeze.)
        try:
            result = subprocess.run(["git", "push", "-q", "origin", branch], cwd=ROOT, capture_output=True, text=True,
                                    timeout=120)
            if result.returncode != 0:
                print(f"warning: push of {branch} failed: {result.stderr.strip()}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"warning: push of {branch} timed out after 120 s; it will be retried after the next generation",
                  flush=True)

    # --- start / resume ---------------------------------------------------------------------------
    def start(self, seed_path):
        if self.dir.exists():
            raise FileExistsError(f"{self.dir} exists; use --resume {self.run_id}")
        for sub in ("candidates", "llm", "scores/evo", "logs"):
            self.path(sub).mkdir(parents=True)
        seed_code = Path(seed_path).read_text()
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        write_json_atomic(self.path("config.json"), {**self.config, "run_id": self.run_id, "git_commit": commit,
                                                     "seed_program": str(seed_path), "seed_sha256": sha256(seed_code)})
        problems = check_program(seed_code, self.config["model"])
        if problems:
            raise ValueError(f"seed program fails the static checks: {problems}")
        self.path("candidates", "candidate_seed.py").write_text(seed_code)
        stage_scores, reached, status, reason = self.cascade(self.path("candidates", "candidate_seed.py"), "seed",
                                                             self.dir, self.config)
        if status != "ok":
            raise RuntimeError(f"seed evaluation failed: {reason}")
        record = {"parent_id": None, "generation": 0, "code_sha256": sha256(seed_code), "status": status,
                  "stage_reached": reached, "scores": stage_scores}
        for island in range(self.config["database"]["num_islands"]):
            self.db.add("seed", record, island)
        self.event("seed", id="seed", scores=stage_scores)
        self.generation = 0
        self.checkpoint()
        self.push()

    def resume(self):
        data = json.loads(self.path("checkpoint.json").read_text())
        self.db = Database.from_dict(data, self.config["database"], self.rng)
        self.rng.setstate(rng_state_from_json(data["rng_state"]))
        self.generation = data["generation"]
        self.event("resume", from_generation=self.generation)

    # --- one child --------------------------------------------------------------------------------
    def child(self, generation, slot):
        cid = f"g{generation:03d}_c{slot:02d}"
        # Rotate islands by the global child index, so runs with few children per generation still use every island.
        index = (generation - 1) * self.config["children_per_generation"] + slot
        island = index % self.config["database"]["num_islands"]
        parent_id, inspiration_ids = self.db.sample(island, self.config["num_inspirations"])
        parent = self.db.programs[parent_id]
        parent_code = self.path("candidates", f"candidate_{parent_id}.py").read_text()
        inspirations = [(self.path("candidates", f"candidate_{i}.py").read_text(), self.db.full_scores(i))
                        for i in inspiration_ids]
        full_n = self.config["cascade"]["stages"][-1]["n"]
        prompt, settings = self.builder.build(parent_code, self.db.full_scores(parent_id),
                                              scores_path(self.dir, parent_id, full_n, full=True), inspirations,
                                              tried=self.tried_from(parent_id))
        record = {"parent_id": parent_id, "inspirations": inspiration_ids, "generation": generation, "scores": {}}

        # Draw the model and temperature even when the call is cached, so a resumed run consumes the random
        # sequence exactly like the original one and picks the same parents afterwards.
        ensemble = self.config["llm_ensemble"]
        member = self.rng.choices(ensemble, weights=[m["weight"] for m in ensemble])[0]
        temperature = self.rng.choice(self.config["llm_temperatures"])
        record["model"] = member["model"]
        llm_file = self.path("llm", f"{cid}.json")
        if llm_file.exists():                                    # resume: never pay for the same call twice
            call = json.loads(llm_file.read_text())
        else:
            if self.path(PAUSE_FILE).exists():
                raise Paused(f"{PAUSE_FILE} file found in the run directory")
            try:
                text, meta = self.llm.generate(self.builder.system(), prompt, member, temperature,
                                               on_failure=lambda failure: self.log_api_failure(generation, cid, failure))
            except APIUnavailable as e:
                raise Paused(f"API returned {e.status}: {e.message}", status=e.status) from e
            except CallFailed as e:
                return cid, island, {**record, "status": "llm_error", "stage_reached": None, "reason": str(e)}
            call = {"id": cid, "prompt": prompt, "settings": settings, "response": text, **meta}
            write_json_atomic(llm_file, call)

        try:
            code, kind = apply_response(parent_code, call["response"])
        except DiffError as e:
            return cid, island, {**record, "status": "invalid_diff", "stage_reached": None, "reason": str(e)}
        record.update(code_sha256=sha256(code), change=kind)
        self.path("candidates", f"candidate_{cid}.py").write_text(code)
        problems = check_program(code, self.config["model"])
        if problems:
            return cid, island, {**record, "status": "rejected", "stage_reached": "static", "reason": problems}
        if any(p.get("code_sha256") == record["code_sha256"] for p in self.db.programs.values()):
            return cid, island, {**record, "status": "duplicate", "stage_reached": "static"}
        sibling, similarity = self.same_change_as_sibling(parent_id, parent_code, code)
        if sibling:
            return cid, island, {**record, "status": "near_duplicate", "stage_reached": "static",
                                 "reason": f"same edit as sibling {sibling} (similarity {similarity:.2f})"}
        stage_scores, reached, status, reason = self.cascade(
            self.path("candidates", f"candidate_{cid}.py"), cid, self.dir, self.config,
            parent_subset=parent["scores"].get("subset"))
        return cid, island, {**record, "status": status, "stage_reached": reached, "scores": stage_scores,
                             "reason": reason}

    def tried_from(self, parent_id, limit=8):
        """(idea, outcome) for the latest earlier children of `parent_id`: the first line of each LLM reply and what
        happened to the child. The pilot and the full run showed the LLM repeating the same edit for the same parent."""
        parent_acc = self.db.full_scores(parent_id)["accuracy"]
        tried = []
        for pid, program in self.db.programs.items():
            llm_file = self.path("llm", f"{pid}.json")
            if program.get("parent_id") != parent_id or not llm_file.exists():
                continue
            idea = json.loads(llm_file.read_text())["response"].strip().split("\n")[0][:160]
            full = program.get("scores", {}).get("full")
            if full:
                delta = full["accuracy"] - parent_acc
                outcome = "same accuracy as the current program" if abs(delta) < 1e-9 else f"{delta:+.1%} accuracy"
            else:
                outcome = {"stopped": "stopped: worse on the 64-query check", "near_duplicate": "a repeat",
                           "duplicate": "a repeat", "invalid_diff": "edit could not be applied"}.get(
                               program["status"], program["status"])
            tried.append((idea, outcome))
        return tried[-limit:]

    def same_change_as_sibling(self, parent_id, parent_code, code):
        """(sibling id, similarity) if another child of the same parent already made essentially this edit, else
        (None, 0). Only siblings are compared, so recombining ideas from different parents is never blocked."""
        threshold = self.config.get("near_duplicate_similarity", 0.8)
        mine = change_set(parent_code, code)
        for pid, program in self.db.programs.items():
            path = self.path("candidates", f"candidate_{pid}.py")
            if program.get("parent_id") != parent_id or not path.exists():
                continue
            similarity = change_similarity(mine, change_set(parent_code, path.read_text()))
            if similarity >= threshold:
                return pid, similarity
        return None, 0.0

    def log_api_failure(self, generation, cid, failure):
        """Append one failed API attempt to population/<run_id>/api_errors.jsonl."""
        with open(self.path("api_errors.jsonl"), "a") as f:
            f.write(json.dumps({"time": now(), "run_id": self.run_id, "generation": generation, "child": cid,
                                **failure}) + "\n")

    def run(self, generations):
        """Run up to `generations`. Returns {"status": "done"} or {"status": "paused", "reason", "api_status"}.

        On a pause the checkpoint is NOT rewritten: it still describes the last completed generation (with its random
        state), so a resume redoes the unfinished generation identically, reusing its cached LLM calls and scores.
        The cached work is pushed so it survives the machine."""
        try:
            self._run(generations)
        except Paused as p:
            self.event("paused", generation=self.generation + 1, reason=p.reason, api_status=p.status)
            self.push()
            return {"status": "paused", "reason": p.reason, "api_status": p.status}
        return {"status": "done"}

    def _run(self, generations):
        for generation in range(self.generation + 1, generations + 1):
            for slot in range(self.config["children_per_generation"]):
                cid, island, record = self.child(generation, slot)
                self.db.add(cid, record, island)
                full = record.get("scores", {}).get("full")
                self.event("child", id=cid, island=island, parent=record["parent_id"], model=record.get("model"),
                           status=record["status"],
                           stage=record["stage_reached"], reason=record.get("reason"),
                           accuracy=full["accuracy"] if full else None)
                print(f"[gen {generation} {cid}] {record['status']:<12} stage={record['stage_reached']} "
                      f"{'acc=%.3f' % full['accuracy'] if full else ''} {record.get('reason') or ''}", flush=True)
            if generation % self.config["database"]["migration_interval"] == 0:
                self.db.migrate()
                self.event("migration", generation=generation)
            self.generation = generation
            self.checkpoint()
            self.push()
            best = self.db.best()
            print(f"== generation {generation} done; best {best}; LLM cost ${self.usage()['cost_usd']:.3f}", flush=True)

    def export(self, which, target):
        """Copy the best program for `which` (accuracy|speed|overall) to a top-level deliverable file."""
        pid = self.db.best()[which]
        shutil.copy(self.path("candidates", f"candidate_{pid}.py"), ROOT / target)
        return pid
