"""Mutation LLM: Gemini models through ArvanCloud's OpenAI-compatible gateway (the loop samples the model per call).

The key is read from ARVAN_API_KEY, e.g. from a git-ignored `.env` at the repo root. The gateway expects the header
`Authorization: apikey <KEY>` (not `Bearer`). Each call returns the text plus a usage record; the loop stores both in
`llm/<id>.json`, which doubles as the cache on resume (a child whose record exists is never sent to the LLM again).
"""
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://api.arvancloudai.ir/v1"
DEFAULT_MAX_TOKENS = 4096

# Transient: retry with exponential backoff. Account/model problems: pause the run, no retry.
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
PAUSE_STATUS = {401, 402, 403, 404}  # bad key, no credit, forbidden, model retired or unknown (gateway: 404)


class APIUnavailable(RuntimeError):
    """The API cannot be used now (account, key, model, or transient errors that outlasted every retry): pause."""
    def __init__(self, status, message):
        super().__init__(f"{status}: {message}")
        self.status, self.message = status, message


class CallFailed(RuntimeError):
    """This one request was rejected (e.g. 400 for this prompt); the run continues with the next child."""


def load_dotenv(path=ROOT / ".env"):
    """Minimal .env reader (KEY=value lines); existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class LLM:
    def __init__(self, max_retries=6, timeout=300, backoff=10):
        load_dotenv()
        from openai import OpenAI
        key = os.environ.get("ARVAN_API_KEY")
        if not key:
            raise RuntimeError("ARVAN_API_KEY is not set (put it in .env at the repo root; see README.md)")
        # The OpenAI client always sends "Bearer <api_key>"; the custom header replaces it with the gateway's scheme.
        self.client = OpenAI(base_url=BASE_URL, api_key=key, default_headers={"Authorization": f"apikey {key}"},
                             timeout=timeout, max_retries=0)
        self.max_retries, self.backoff = max_retries, backoff

    def generate(self, system, prompt, member, temperature, on_failure=None):
        """(text, record) from `member` = {"model", "price_in", "price_out"} (USD per million tokens, thinking tokens
        billed as output), optionally "reasoning_effort" and "max_tokens". Retries transient errors with backoff.

        `max_tokens` (default 4096) caps all output tokens, thinking included, so no single call can run away in cost;
        a reply cut off by it ends with finish_reason "length" and is rejected by the diff parser.

        Errors: transient ones (429, 5xx, timeouts) are retried with exponential backoff (10 s, 20 s, ... up to
        `max_retries` attempts); account and model errors (401/402/403/404) are retried once after the backoff and,
        if they repeat, raise APIUnavailable (the run pauses), as do retries that run out; other client errors
        (e.g. 400) raise CallFailed for this call only.
        `on_failure(dict)` is called for every failed attempt (for the run's API error log)."""
        extra = {"reasoning_effort": member["reasoning_effort"]} if member.get("reasoning_effort") else {}
        extra["max_tokens"] = member.get("max_tokens", DEFAULT_MAX_TOKENS)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        confirmed_pause = False
        for attempt in range(1, self.max_retries + 1):
            start = time.time()
            try:
                response = self.client.chat.completions.create(model=member["model"], messages=messages,
                                                               temperature=temperature, **extra)
                break
            except Exception as e:
                status, kind = classify(e)
                if on_failure:
                    on_failure({"model": member["model"], "status": status, "kind": kind, "attempt": attempt,
                                "error": type(e).__name__, "message": str(e)[:300]})
                if kind == "pause":
                    # Confirm once before pausing: the gateway has returned a spurious 401 that succeeded on retry.
                    if confirmed_pause or attempt == self.max_retries:
                        raise APIUnavailable(status, str(e)[:300]) from e
                    confirmed_pause = True
                if kind == "call":
                    raise CallFailed(f"{status}: {str(e)[:300]}") from e
                if attempt == self.max_retries:
                    raise APIUnavailable(status, f"still failing after {attempt} attempts: {str(e)[:250]}") from e
                wait = min(self.backoff * 2 ** (attempt - 1), 300)
                print(f"LLM error {status} ({type(e).__name__}); retry {attempt}/{self.max_retries - 1} in {wait}s",
                      flush=True)
                time.sleep(wait)
        usage = response.usage
        tokens_in = getattr(usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(usage, "completion_tokens", 0) or 0  # includes reasoning tokens
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = (getattr(details, "reasoning_tokens", 0) or 0) if details else 0
        record = {
            "model": member["model"], "temperature": temperature, "latency": time.time() - start,
            "input_tokens": tokens_in, "output_tokens": tokens_out, "reasoning_tokens": reasoning,
            "finish_reason": response.choices[0].finish_reason,
            "cost_usd": (tokens_in * member.get("price_in", 0) + tokens_out * member.get("price_out", 0)) / 1e6,
        }
        return response.choices[0].message.content or "", record

    def ping(self, member):
        """One minimal call to check that `member`'s model answers (used before resuming a paused run)."""
        return self.generate("Reply with one word.", "Say OK.", dict(member, max_tokens=16), 0.0)


def classify(error):
    """(HTTP status or error name, "retry" | "pause" | "call") for an exception from the OpenAI client."""
    import openai
    if isinstance(error, (openai.APITimeoutError, openai.APIConnectionError)):
        return type(error).__name__, "retry"
    status = getattr(error, "status_code", None)
    if status in PAUSE_STATUS:
        return status, "pause"
    if status in RETRY_STATUS or (status is not None and status >= 500):
        return status, "retry"
    if status is not None:
        return status, "call"
    return type(error).__name__, "retry"  # unknown network-level problem: treat as transient
