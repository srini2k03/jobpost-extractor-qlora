"""OpenAI-compatible (Gemini free tier) runner with a mocked client: no network, no key."""

import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

openai = pytest.importorskip("openai")
import httpx  # noqa: E402

from jobx.baselines import openai_compat_runner as oc  # noqa: E402
from jobx.data.build import read_jsonl  # noqa: E402

REQ = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")


def completion(text: str, model: str = "gemini-2.5-flash", pin: int = 900, pout: int = 120):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=pin, completion_tokens=pout),
        model=model,
    )


def rate_limit(retry_after: str | None = "1"):
    headers = {"retry-after": retry_after} if retry_after else {}
    return openai.RateLimitError(
        "429 quota", response=httpx.Response(429, headers=headers, request=REQ), body=None
    )


class FakeClient:
    def __init__(self, handler):
        self.calls: list[dict] = []

        async def create(**kw):
            self.calls.append(kw)
            return handler(kw, len(self.calls))

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


async def no_sleep(_: float) -> None:
    return None


def label_for(kw: dict, examples: list[dict]) -> dict:
    text = kw["messages"][-1]["content"]
    return next(e for e in examples if f"<posting>\n{e['text'].strip()}\n</posting>" in text)["label"]


def run(client, examples, tmp_path, **kw):
    limiter = oc.RateLimiter(0, sleep=no_sleep)
    defaults = dict(
        base_url="https://x/",
        model="gemini-2.5-flash",
        approach_prefix="gemini-flash",
        shots=[],
        split="test_synth",
        out_dir=tmp_path / "pred",
        limiter=limiter,
        cache=oc.ResponseCache(tmp_path / "cache"),
        sleep=no_sleep,
    )
    defaults.update(kw)
    return asyncio.run(oc.run_split(client, examples, **defaults))


def test_request_uses_shared_prompt_and_json_schema():
    kw = oc.build_request("gemini-2.5-flash", "posting", [], mode="json_schema", max_tokens=512)
    assert kw["messages"][0]["role"] == "system" and kw["temperature"] == 0
    rf = kw["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["name"] == "JobPosting"
    s = json.dumps(rf["json_schema"]["schema"])
    assert "$ref" not in s and "minimum" not in s and "company" in s
    assert oc.build_request("m", "p", [], mode="json_object", max_tokens=1)["response_format"] == {
        "type": "json_object"
    }
    assert "response_format" not in oc.build_request("m", "p", [], mode="none", max_tokens=1)
    assert oc.PROVIDERS["gemini"].base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert oc.PROVIDERS["gemini"].key_env == "GEMINI_API_KEY"


def test_rate_limiter_spaces_requests():
    t = {"now": 0.0}
    slept: list[float] = []

    async def sleep(d: float) -> None:
        slept.append(d)
        t["now"] += d

    lim = oc.RateLimiter(10, clock=lambda: t["now"], sleep=sleep)

    async def go():
        for _ in range(3):
            await lim.wait()

    asyncio.run(go())
    assert slept == [6.0, 6.0]


def test_retry_on_429_then_quota_exhausted():
    lim = oc.RateLimiter(0, sleep=no_sleep)
    errs = [rate_limit("2")]

    async def create(**kw):
        if errs:
            raise errs.pop(0)
        return "ok"

    assert asyncio.run(oc.call_with_retries(create, {}, limiter=lim, sleep=no_sleep)) == "ok"

    async def always(**kw):
        raise rate_limit(None)

    with pytest.raises(oc.QuotaExhausted):
        asyncio.run(oc.call_with_retries(always, {}, limiter=lim, sleep=no_sleep, max_attempts=3))


def test_run_writes_predictions_tokens_and_zero_cost(tmp_path):
    examples = read_jsonl("data/processed/test_synth.jsonl")[:5]
    client = FakeClient(lambda kw, n: completion(json.dumps(label_for(kw, examples))))
    pred, st = run(client, examples, tmp_path)
    rows = [json.loads(x) for x in pred.read_text().splitlines()]
    assert [r["id"] for r in rows] == [e["id"] for e in examples]
    assert st.n_new == 5 and st.prompt_tokens == 4500 and st.completion_tokens == 600 and st.cost_usd == 0
    meta = json.loads(pred.with_name("gemini-flash_zero_shot__test_synth.meta.json").read_text())
    assert meta["usage"] == {"requests": 5, "prompt_tokens": 4500, "completion_tokens": 600}
    assert meta["cost_usd"] == 0 and meta["complete"] and meta["response_format"] == "json_schema"
    assert "key" not in json.dumps(meta).lower()


def test_format_downgrade_on_400(tmp_path):
    examples = read_jsonl("data/processed/test_synth.jsonl")[:2]

    def handler(kw, n):
        if kw.get("response_format", {}).get("type") == "json_schema":
            raise openai.BadRequestError(
                "Invalid JSON payload: response_format json_schema not supported",
                response=httpx.Response(400, request=REQ),
                body=None,
            )
        return completion(json.dumps(label_for(kw, examples)))

    _, st = run(FakeClient(handler), examples, tmp_path, concurrency=1)
    assert st.response_format == "json_object" and st.format_downgrades == ["json_schema->json_object"]
    assert st.n_errors == 0 and st.n_new == 2


def test_cache_and_resume_after_quota(tmp_path):
    examples = read_jsonl("data/processed/test_synth.jsonl")[:6]
    state = {"ok": 3}

    def flaky(kw, n):
        if state["ok"] <= 0:
            raise rate_limit(None)
        state["ok"] -= 1
        return completion(json.dumps(label_for(kw, examples)))

    c1 = FakeClient(flaky)
    pred, st = run(c1, examples, tmp_path, concurrency=1)
    assert st.quota_exhausted and len(pred.read_text().splitlines()) == 3
    meta = json.loads(pred.with_name(pred.name.replace(".jsonl", ".meta.json")).read_text())
    assert not meta["complete"]
    # next day: quota back; finished ids are skipped, only the remaining 3 are requested
    c2 = FakeClient(lambda kw, n: completion(json.dumps(label_for(kw, examples))))
    pred, st2 = run(c2, examples, tmp_path, concurrency=1)
    assert st2.n_resumed == 3 and st2.n_new == 3 and len(c2.calls) == 3
    assert [json.loads(x)["id"] for x in pred.read_text().splitlines()] == [e["id"] for e in examples]
    # a fresh predictions dir with the same cache: all served from cache, no requests
    c3 = FakeClient(lambda kw, n: pytest.fail("should be cached"))
    _, st3 = run(c3, examples, tmp_path, out_dir=tmp_path / "pred2")
    assert st3.n_cached == 6 and not c3.calls


def test_main_end_to_end_and_missing_key(tmp_path, monkeypatch, capsys):
    root = tmp_path / "root"
    (root / "data/processed").mkdir(parents=True)
    for f in ("train.jsonl", "test_synth.jsonl"):
        shutil.copy(Path("data/processed") / f, root / "data/processed" / f)
    examples = read_jsonl(root / "data/processed/test_synth.jsonl")
    client = FakeClient(lambda kw, n: completion(json.dumps(label_for(kw, examples))))
    args = oc.parser().parse_args(
        ["--root", str(root), "--max-examples", "4", "--rpm", "0", "--cache-dir", str(tmp_path / "c")]
    )
    assert asyncio.run(oc.main_async(args, client=client)) == 0
    assert len(client.calls[0]["messages"]) == 8  # system + 3 shots x 2 + posting
    m = json.loads((root / "results/metrics/gemini-flash_few_shot__test_synth.json").read_text())
    assert m["metrics"]["micro_f1"] == pytest.approx(1.0) and m["metrics"]["n"] == 4

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert asyncio.run(oc.main_async(oc.parser().parse_args(["--root", str(root)]))) == 2
    assert "GEMINI_API_KEY is not set" in capsys.readouterr().err
