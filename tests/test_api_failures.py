"""API failure handling: retry vs. pause classification, the run's error log, clean pause, PAUSE file, resume."""
import json
import shutil
import sys
from pathlib import Path

import httpx
import openai
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import alphaevolve.llm as llm_module  # noqa: E402
import alphaevolve.loop as loop  # noqa: E402
from alphaevolve.llm import LLM, APIUnavailable, CallFailed  # noqa: E402
from test_loop import CONFIG, SEED, FakeBuilder, FakeLLM, fake_cascade  # noqa: E402

MEMBER = {"model": "m", "price_in": 0.25, "price_out": 1.5}


def status_error(status):
    response = httpx.Response(status, request=httpx.Request("POST", "https://gateway/v1/chat/completions"))
    cls = {400: openai.BadRequestError, 401: openai.AuthenticationError, 403: openai.PermissionDeniedError,
           404: openai.NotFoundError, 429: openai.RateLimitError}.get(status, openai.APIStatusError)
    return cls(f"error {status}", response=response, body=None)


class Reply:
    """Minimal stand-in for an OpenAI chat completion."""
    def __init__(self, text):
        message = type("M", (), {"content": text})
        self.choices = [type("C", (), {"message": message, "finish_reason": "stop"})]
        self.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 2, "completion_tokens_details": None})


def scripted_llm(outcomes, monkeypatch):
    """An LLM whose client raises or answers according to `outcomes` (exceptions or reply texts), in order."""
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)
    llm = LLM.__new__(LLM)
    llm.max_retries, llm.backoff = 4, 10
    queue = list(outcomes)

    def create(**kwargs):
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return Reply(outcome)
    llm.client = type("Client", (), {"chat": type("Chat", (), {"completions": type("Comp", (), {
        "create": staticmethod(create)})})})
    return llm, queue


def test_rate_limit_and_outage_are_retried(monkeypatch):
    failures = []
    llm, queue = scripted_llm([status_error(429), status_error(503), openai.APITimeoutError(
        request=httpx.Request("POST", "https://x")), "OK"], monkeypatch)
    text, record = llm.generate("s", "p", MEMBER, 0.0, on_failure=failures.append)
    assert text == "OK" and not queue
    assert [f["status"] for f in failures] == [429, 503, "APITimeoutError"]
    assert all(f["kind"] == "retry" for f in failures)


@pytest.mark.parametrize("status", [401, 402, 403, 404])
def test_account_and_model_errors_pause_after_one_confirmation(monkeypatch, status):
    failures = []
    llm, queue = scripted_llm([status_error(status), status_error(status), "never reached"], monkeypatch)
    with pytest.raises(APIUnavailable) as info:
        llm.generate("s", "p", MEMBER, 0.0, on_failure=failures.append)
    assert info.value.status == status and len(failures) == 2 and queue == ["never reached"]


def test_spurious_401_does_not_pause(monkeypatch):
    llm, queue = scripted_llm([status_error(401), "OK"], monkeypatch)
    assert llm.generate("s", "p", MEMBER, 0.0)[0] == "OK" and not queue


def test_retries_that_run_out_pause(monkeypatch):
    llm, _ = scripted_llm([status_error(429)] * 4, monkeypatch)
    with pytest.raises(APIUnavailable, match="still failing after 4 attempts"):
        llm.generate("s", "p", MEMBER, 0.0)


def test_bad_request_fails_only_that_call(monkeypatch):
    llm, _ = scripted_llm([status_error(400)], monkeypatch)
    with pytest.raises(CallFailed):
        llm.generate("s", "p", MEMBER, 0.0)


class FlakyLLM(FakeLLM):
    """FakeLLM that loses its credit (402) on call number `fail_at`, logging the failure like the real client."""
    def __init__(self, fail_at):
        super().__init__()
        self.fail_at = fail_at

    def generate(self, system, prompt, member, temperature, on_failure=None):
        if self.calls + 1 == self.fail_at:
            if on_failure:
                on_failure({"model": member["model"], "status": 402, "kind": "pause", "attempt": 1})
            raise APIUnavailable(402, "insufficient credit")
        return super().generate(system, prompt, member, temperature)


def make(tmp_path, monkeypatch, llm):
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    evo = loop.Evolution(CONFIG, "instruct_test", prompt_builder=FakeBuilder(), cascade=fake_cascade, push=False)
    evo.llm = llm
    return evo


def reference_run(tmp_path, monkeypatch):
    seed = tmp_path / "seed.py"
    seed.write_text(SEED)
    evo = make(tmp_path / "ref", monkeypatch, FakeLLM())
    (tmp_path / "ref").mkdir()
    evo.start(seed)
    evo.run(generations=3)
    return {k: v["parent_id"] for k, v in evo.db.programs.items()}


def test_api_pause_then_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    expected = reference_run(tmp_path, monkeypatch)
    seed = tmp_path / "seed.py"
    evo = make(tmp_path, monkeypatch, FlakyLLM(fail_at=7))     # 4 children/generation: fails in generation 2
    evo.start(seed)
    result = evo.run(generations=3)
    assert result == {"status": "paused", "reason": "API returned 402: insufficient credit", "api_status": 402}
    ckpt = json.loads((evo.dir / "checkpoint.json").read_text())
    assert ckpt["generation"] == 1                              # last completed generation, untouched
    errors = [json.loads(line) for line in (evo.dir / "api_errors.jsonl").read_text().splitlines()]
    assert errors[0]["status"] == 402 and errors[0]["generation"] == 2 and errors[0]["run_id"] == "instruct_test"
    assert "paused" in (evo.dir / "events.jsonl").read_text()

    resumed = make(tmp_path, monkeypatch, FakeLLM())            # credit topped up
    resumed.resume()
    assert resumed.run(generations=3) == {"status": "done"}
    # Same selections as a run that never paused (the fake LLM's replies depend on its own call counter, so
    # statuses after the pause can differ; parent choice is what the resume must reproduce).
    assert {k: v["parent_id"] for k, v in resumed.db.programs.items()} == expected
    assert resumed.llm.calls == 12 - 6                          # generation 2's two finished calls were reused


def test_pause_file_stops_before_the_next_llm_call(tmp_path, monkeypatch):
    seed = tmp_path / "seed.py"
    seed.write_text(SEED)
    llm = FakeLLM()
    evo = make(tmp_path, monkeypatch, llm)
    evo.start(seed)
    (evo.dir / loop.PAUSE_FILE).touch()
    result = evo.run(generations=2)
    assert result["status"] == "paused" and result["api_status"] is None and llm.calls == 0
    assert json.loads((evo.dir / "checkpoint.json").read_text())["generation"] == 0
    (evo.dir / loop.PAUSE_FILE).unlink()                        # what --resume does
    resumed = make(tmp_path, monkeypatch, FakeLLM())
    resumed.resume()
    assert resumed.run(generations=2) == {"status": "done"} and resumed.generation == 2
    shutil.rmtree(evo.dir)
