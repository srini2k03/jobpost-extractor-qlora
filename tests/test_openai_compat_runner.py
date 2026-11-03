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


def completion(text: str, model: str = "gemini-3.8-flash", pin: int = 900, pout: int = 120):
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
        model="gemini-3.8-flash",
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
    kw = oc.build_request("gemini-3.8-flash", "posting", [], mode="json_schema", max_tokens=512)
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
    groq = oc.PROVIDERS["groq"]
    assert oc.DEFAULT_PROVIDER == "groq" and oc.parser().parse_args([]).provider == "groq"
    assert groq.base_url == "https://api.groq.com/openai/v1" and groq.key_env == "GROQ_API_KEY"
    assert groq.default_model == "llama-3.3-70b-versatile" and groq.default_rpm == 25
    assert groq.default_fallbacks == ("openai/gpt-oss-120b", "qwen/qwen3-32b")
    assert groq.approach_prefix == "llama-3.3-70b-groq"


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
    m = json.loads((root / "results/metrics/llama-3.3-70b-groq_few_shot__test_synth.json").read_text())
    assert m["metrics"]["micro_f1"] == pytest.approx(1.0) and m["metrics"]["n"] == 4

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert asyncio.run(oc.main_async(oc.parser().parse_args(["--root", str(root)]))) == 2
    assert "GROQ_API_KEY is not set" in capsys.readouterr().err
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert (
        asyncio.run(oc.main_async(oc.parser().parse_args(["--root", str(root), "--provider", "gemini"]))) == 2
    )
    assert "GEMINI_API_KEY is not set" in capsys.readouterr().err


def status_error(cls, code: int, msg: str):
    return cls(msg, response=httpx.Response(code, request=REQ), body=None)


def test_fatal_401_aborts_without_writing(tmp_path):
    examples = read_jsonl("data/processed/test_synth.jsonl")[:3]

    def handler(kw, n):
        raise status_error(openai.AuthenticationError, 401, "API key not valid")

    client = FakeClient(handler)
    with pytest.raises(oc.FatalConfigError, match="401"):
        run(client, examples, tmp_path)
    assert len(client.calls) == 1  # not retried
    assert not (tmp_path / "pred").exists() or not list((tmp_path / "pred").iterdir())
    assert not list((tmp_path / "cache").iterdir())


def test_404_on_only_model_is_fatal_and_main_exits_nonzero(tmp_path, capsys):
    root = tmp_path / "root"
    (root / "data/processed").mkdir(parents=True)
    for f in ("train.jsonl", "test_synth.jsonl"):
        shutil.copy(Path("data/processed") / f, root / "data/processed" / f)
    client = FakeClient(
        lambda kw, n: (_ for _ in ()).throw(
            status_error(openai.NotFoundError, 404, "model no longer available to new users")
        )
    )
    args = oc.parser().parse_args(
        [
            "--root",
            str(root),
            "--max-examples",
            "3",
            "--rpm",
            "0",
            "--model-fallbacks",
            "",
            "--cache-dir",
            str(tmp_path / "c"),
        ]
    )
    assert asyncio.run(oc.main_async(args, client=client)) == oc.EXIT_FATAL
    assert "no longer available" in capsys.readouterr().err
    assert not (root / "results/predictions").exists() or not list(
        (root / "results/predictions").glob("*.jsonl")
    )
    assert not (root / "results/metrics").exists()


def test_fallback_after_two_503s_and_404_drops_model(tmp_path):
    examples = read_jsonl("data/processed/test_synth.jsonl")[:4]

    def handler(kw, n):
        if kw["model"] == "llama-3.3-70b-versatile":
            raise status_error(openai.InternalServerError, 503, "model currently overloaded")
        if kw["model"] == "openai/gpt-oss-120b":
            raise status_error(openai.NotFoundError, 404, "not found")
        return completion(json.dumps(label_for(kw, examples)), model=kw["model"])

    client = FakeClient(handler)
    chain = oc.ModelChain(
        ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "qwen/qwen3-32b"], cooldown_s=1e9
    )
    pred, st = run(client, examples, tmp_path, model=chain, concurrency=1)
    tried = [c["model"] for c in client.calls]
    assert tried[:4] == [
        "llama-3.3-70b-versatile",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        "qwen/qwen3-32b",
    ]
    assert tried.count("llama-3.3-70b-versatile") == 2  # cooled down, not retried for later examples
    rows = [json.loads(x) for x in pred.read_text().splitlines()]
    assert {r["model"] for r in rows} == {"qwen/qwen3-32b"} and st.n_errors == 0
    meta = json.loads(pred.with_name(pred.name.replace(".jsonl", ".meta.json")).read_text())
    assert meta["models_used"] == {"qwen/qwen3-32b": 4}
    assert list(meta["dropped_models"]) == ["openai/gpt-oss-120b"]
    assert meta["model"] == "llama-3.3-70b-versatile" and meta["model_chain"][0] == "llama-3.3-70b-versatile"


def test_primary_comes_back_after_cooldown():
    t = {"now": 0.0}
    chain = oc.ModelChain(["a", "b"], switch_after=2, cooldown_s=60, clock=lambda: t["now"])
    assert chain.current() == ("a", 0.0)
    chain.record_overload("a")
    assert chain.current()[0] == "a"
    chain.record_overload("a")
    assert chain.current() == ("b", 0.0)
    t["now"] = 61
    assert chain.current() == ("a", 0.0)
    chain.drop("a", "404")
    chain.drop("b", "403")
    with pytest.raises(oc.FatalConfigError, match="every model"):
        chain.current()


def test_errors_are_never_cached(tmp_path):
    cache = oc.ResponseCache(tmp_path / "c")
    cache.put("k1", {"output": "", "error": "NotFoundError"})
    cache.put("k2", {"output": "   "})
    assert not list((tmp_path / "c").iterdir())
    # an error entry left by an older version is ignored on read
    (tmp_path / "c" / "k3.json").write_text(json.dumps({"output": "", "error": "NotFoundError"}))
    assert cache.get("k3") is None
    cache.put("k4", {"output": '{"company": null}'})
    assert cache.get("k4") == {"output": '{"company": null}'}

    examples = read_jsonl("data/processed/test_synth.jsonl")[:2]

    def bad(kw, n):
        raise status_error(openai.BadRequestError, 400, "content filtered")

    run(FakeClient(bad), examples, tmp_path, mode="none")
    assert not list((tmp_path / "cache").glob("*.json"))


def test_all_requests_failed_writes_nothing(tmp_path):
    examples = read_jsonl("data/processed/test_synth.jsonl")[:3]
    pred, st = run(
        FakeClient(
            lambda kw, n: (_ for _ in ()).throw(status_error(openai.BadRequestError, 400, "content filtered"))
        ),
        examples,
        tmp_path,
        mode="none",
    )
    assert pred is None and st.n_completed == 0 and st.n_errors == 3
    assert not list((tmp_path / "pred").glob("*.jsonl"))


def test_groq_rate_limit_headers():
    assert oc.parse_duration("7") == 7 and oc.parse_duration("350ms") == pytest.approx(0.35)
    assert oc.parse_duration("2m59.56s") == pytest.approx(179.56) and oc.parse_duration("1h2m") == 3720
    assert oc.parse_duration("soon") is None
    slept: list[float] = []

    async def sleep(d: float) -> None:
        slept.append(d)

    err = openai.RateLimitError(
        "429",
        response=httpx.Response(429, headers={"x-ratelimit-reset-requests": "2.5s"}, request=REQ),
        body=None,
    )
    errs = [err]

    async def create(**kw):
        if errs:
            raise errs.pop(0)
        return "ok"

    lim = oc.RateLimiter(0, sleep=no_sleep)
    assert asyncio.run(oc.call_with_retries(create, {}, limiter=lim, sleep=sleep)) == "ok"
    assert 2.5 <= slept[0] < 3.5


def test_preset_defaults_resolve_rpm_and_prefix(tmp_path, capsys):
    root = tmp_path / "root"
    (root / "data/processed").mkdir(parents=True)
    for f in ("train.jsonl", "test_synth.jsonl"):
        shutil.copy(Path("data/processed") / f, root / "data/processed" / f)
    examples = read_jsonl(root / "data/processed/test_synth.jsonl")
    client = FakeClient(lambda kw, n: completion(json.dumps(label_for(kw, examples)), model=kw["model"]))
    args = oc.parser().parse_args(
        ["--root", str(root), "--max-examples", "2", "--cache-dir", str(tmp_path / "c")]
    )
    assert args.rpm is None and args.approach_prefix is None
    # the limiter would wait 2.4 s between requests at 25 RPM; patch it out to keep the test fast
    orig = oc.RateLimiter.__init__

    def fast(self, rpm, **kw):
        assert rpm == 25.0
        orig(self, 0, **kw)

    oc.RateLimiter.__init__ = fast  # type: ignore[method-assign]
    try:
        assert asyncio.run(oc.main_async(args, client=client)) == 0
    finally:
        oc.RateLimiter.__init__ = orig  # type: ignore[method-assign]
    out = capsys.readouterr().out
    assert "llama-3.3-70b-versatile -> openai/gpt-oss-120b -> qwen/qwen3-32b" in out
    meta = json.loads(
        (root / "results/predictions/llama-3.3-70b-groq_few_shot__test_synth.meta.json").read_text()
    )
    assert meta["models_used"] == {"llama-3.3-70b-versatile": 2} and meta["provider"] == "groq"
