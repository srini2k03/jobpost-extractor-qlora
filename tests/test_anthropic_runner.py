"""API baseline runner with a mocked async client (no network, no key)."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

from conftest import gold_example  # noqa: E402
from jobx.baselines import anthropic_runner as ar  # noqa: E402
from jobx.data.build import read_jsonl  # noqa: E402
from jobx.eval.run import evaluate_file  # noqa: E402

REQ = httpx2.Request("POST", "https://api.example/v1/messages")


def response(
    text: str, *, model: str = "claude-opus-5-5", stop: str = "end_turn", inp: int = 1500, out: int = 200
):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=inp, output_tokens=out, cache_read_input_tokens=0, cache_creation_input_tokens=0
        ),
        stop_reason=stop,
        model=model,
    )


class FakeMessages:
    def __init__(self, handler):
        self.handler = handler
        self.calls: list[dict] = []

    async def create(self, **kw):
        self.calls.append(kw)
        return self.handler(kw, len(self.calls))

    async def count_tokens(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(input_tokens=1500 if len(kw["messages"]) == 1 else 2100)


class FakeClient:
    def __init__(self, handler):
        self.messages = FakeMessages(handler)
        self.beta = SimpleNamespace(messages=self.messages)


async def no_sleep(_: float) -> None:
    return None


def test_request_shape_uses_shared_prompt_schema_and_fallbacks():
    shots = [{"id": "s1", "text": "Role: Telecaller", "label": gold_example()}]
    kw = ar.build_request("claude-opus-5-5", "posting", shots, effort="low", max_tokens=16000)
    assert kw["system"][0]["text"].startswith("You extract structured data")
    assert [m["role"] for m in kw["messages"]] == ["user", "assistant", "user"]
    assert kw["messages"][1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert kw["messages"][-1]["content"].endswith("</posting>")
    fmt = kw["output_config"]["format"]
    assert fmt["type"] == "json_schema" and "company" in fmt["schema"]["properties"]
    assert kw["output_config"]["effort"] == "low"
    assert kw["fallbacks"] == "default" and kw["betas"] == [ar.FALLBACK_BETA]
    assert "thinking" not in kw  # always on for this model; effort is the control
    kw2 = ar.build_request("claude-haiku-5-5", "p", [], effort="low", max_tokens=100)
    assert "fallbacks" not in kw2 and len(kw2["messages"]) == 1


def test_usage_cost():
    u = ar.Usage(input_tokens=1_000_000, output_tokens=100_000)
    assert u.cost("claude-opus-5-5") == pytest.approx(4.0 + 2.0)
    assert ar.approach_id("claude-opus-5-5", 0) == "claude-opus-5-5_zero_shot"
    assert ar.approach_id("claude-opus-5-5", 3) == "claude-opus-5-5_few_shot"


def test_retries_honour_retry_after_then_succeed():
    slept: list[float] = []

    async def sleep(d: float) -> None:
        slept.append(d)

    errors = [
        anthropic.RateLimitError(
            "rl", response=httpx2.Response(429, headers={"retry-after": "7"}, request=REQ), body=None
        ),
        anthropic.InternalServerError("overloaded", response=httpx2.Response(529, request=REQ), body=None),
    ]

    async def create(**kw):
        if errors:
            raise errors.pop(0)
        return "ok"

    assert asyncio.run(ar.call_with_retries(create, {}, sleep=sleep)) == "ok"
    assert 7 <= slept[0] < 8 and len(slept) == 2


def test_non_retryable_error_raises_immediately():
    calls = []

    async def create(**kw):
        calls.append(1)
        raise anthropic.BadRequestError("bad", response=httpx2.Response(400, request=REQ), body=None)

    with pytest.raises(anthropic.BadRequestError):
        asyncio.run(ar.call_with_retries(create, {}, sleep=no_sleep))
    assert len(calls) == 1


def test_run_split_writes_standard_predictions(tmp_path):
    gold = {e["id"]: e for e in read_jsonl("data/processed/test_synth.jsonl")[:6]}
    examples = list(gold.values())

    def handler(kw, n):
        text = kw["messages"][-1]["content"]
        ex = next(e for e in examples if e["text"].strip() in text)
        if ex is examples[1]:
            return response("", stop="refusal")
        if ex is examples[2]:
            raise anthropic.BadRequestError("nope", response=httpx2.Response(400, request=REQ), body=None)
        return response(
            json.dumps(ex["label"]), model="claude-opus-4-8" if ex is examples[3] else "claude-opus-5-5"
        )

    client = FakeClient(handler)
    budget = ar.Budget(10.0, 0.05)
    pred, s = asyncio.run(
        ar.run_split(
            client,
            examples,
            model="claude-opus-5-5",
            shots=[],
            split="test_synth",
            out_dir=tmp_path,
            budget=budget,
            sleep=no_sleep,
        )
    )
    rows = [json.loads(x) for x in pred.read_text().splitlines()]
    assert [r["id"] for r in rows] == [e["id"] for e in examples]
    assert s.n_done == 6 and s.n_errors == 1 and s.n_refusals == 1 and s.n_fallback_served == 1
    assert rows[2]["error"] == "BadRequestError" and rows[2]["output"] == ""
    assert s.cost_usd == pytest.approx(5 * (1500 * 4 + 200 * 20) / 1e6, rel=0.3)
    meta = json.loads(pred.with_name(pred.name.replace(".jsonl", ".meta.json")).read_text())
    assert meta["approach"] == "claude-opus-5-5_zero_shot" and meta["refusal_fallbacks"] == "default"
    assert "key" not in json.dumps(meta).lower()
    # scored by the normal evaluator
    import shutil

    root = tmp_path / "root"
    (root / "data/processed").mkdir(parents=True)
    shutil.copy("data/processed/test_synth.jsonl", root / "data/processed/test_synth.jsonl")
    res = evaluate_file(pred, root, n_boot=0)
    assert res["metrics"]["n"] == 6 and res["metrics"]["json_validity"] == pytest.approx(4 / 6)


def test_budget_guard_stops_new_requests(tmp_path):
    examples = read_jsonl("data/processed/test_synth.jsonl")[:10]
    client = FakeClient(lambda kw, n: response(json.dumps(gold_example()), inp=10_000, out=10_000))
    budget = ar.Budget(cap_usd=0.5, reserve_per_call=0.1)  # each call really costs 0.24
    _, s = asyncio.run(
        ar.run_split(
            client,
            examples,
            model="claude-opus-5-5",
            shots=[],
            split="test_synth",
            out_dir=tmp_path,
            budget=budget,
            concurrency=1,
            sleep=no_sleep,
        )
    )
    assert s.n_done < 10 and s.skipped_budget == 10 - s.n_done
    assert budget.spent <= 0.5 + 0.24


def test_main_estimates_fits_budget_and_runs(tmp_path, monkeypatch, capsys):
    import shutil

    root = tmp_path
    (root / "data/processed").mkdir(parents=True)
    for f in ("train.jsonl", "test_synth.jsonl"):
        shutil.copy(Path("data/processed") / f, root / "data/processed" / f)
    test = {e["id"]: e for e in read_jsonl(root / "data/processed/test_synth.jsonl")}

    def handler(kw, n):
        ex_text = kw["messages"][-1]["content"]
        ex = next(
            e
            for e in test.values()
            if f"<posting>\n{e['text'].strip()}\n</posting>" == ex_text.split("\n\n", 1)[1]
        )
        return response(json.dumps(ex["label"]))

    client = FakeClient(handler)
    args = ar.parser().parse_args(
        [
            "--root",
            str(root),
            "--max-examples",
            "400",
            "--budget-usd",
            "1.0",
            "--shots",
            "0,3",
            "--splits",
            "test_synth,test_real",
        ]
    )
    assert asyncio.run(ar.main_async(args, client=client)) == 0
    out = capsys.readouterr().out
    est = json.loads(out[: out.index("\n}\n") + 2])
    assert est["total_est_cost_usd"] <= 1.0 and est["fitted_max_examples"] < 400
    m = json.loads((root / "results/metrics/claude-opus-5-5_few_shot__test_synth.json").read_text())
    assert m["metrics"]["micro_f1"] == pytest.approx(1.0)
    assert not (root / "results/predictions/claude-opus-5-5_zero_shot__test_real.jsonl").exists()


def test_missing_key_fails_clearly(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert asyncio.run(ar.main_async(ar.parser().parse_args([]))) == 2
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err
