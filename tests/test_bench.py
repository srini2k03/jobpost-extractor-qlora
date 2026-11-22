"""Serving benchmark harness (jobx.bench), CPU-only against the in-process fake server."""

from __future__ import annotations

import asyncio
import csv
import math
import random
import statistics
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from jobx import prompts
from jobx.bench.cost import gpu_price, usd_per_1k_requests, usd_per_1m_output_tokens
from jobx.bench.fake_server import FakeConfig, FakeServer, running_fake_server
from jobx.bench.gpu import GpuSampler, parse_nvidia_smi
from jobx.bench.loadgen import (
    CSV_FIELDS,
    RunConfig,
    RunResult,
    StreamingClient,
    run_closed_loop,
    run_open_loop,
    sweep,
    write_csv,
)
from jobx.bench.metrics import SLA, markdown_table, read_rows, summarize_rows
from jobx.bench.workload import (
    WORKLOADS,
    BenchRequest,
    OutputLengths,
    build_workload,
    prompt_stats,
    sample_without_replacement,
    shared_prefix_chars,
)
from jobx.cli import app

MATRIX = Path("configs/bench/matrix.yaml")


def _posting(i: int) -> dict:
    return {
        "id": f"p{i:03d}",
        "text": f"Hiring a data engineer {i} in Pune, 3-5 yrs, 12 LPA",
        "label": {},
        "meta": {},
    }


POSTINGS = [_posting(i) for i in range(40)]


def _reqs(n: int, max_tokens: int = 8) -> list[BenchRequest]:
    return build_workload("extract-compact", n, postings=POSTINGS, max_tokens=max_tokens)


# ---------------------------------------------------------------------------------------------- workload


def test_workloads_deterministic_and_shaped() -> None:
    a = build_workload("extract-full", 50, seed=3)
    b = build_workload("extract-full", 50, seed=3)
    c = build_workload("extract-full", 50, seed=4)
    assert [r.messages for r in a] == [r.messages for r in b]
    assert [r.id for r in a] != [r.id for r in c]
    # system + 3 shots (user/assistant) + posting; the 7 leading messages are the shared prefix
    assert all(len(r.messages) == 8 and r.messages[0]["content"] == prompts.SYSTEM_PROMPT for r in a)
    assert shared_prefix_chars(a) > len(prompts.SYSTEM_PROMPT) + 3 * 300
    assert all(r.max_tokens == 384 and not r.extra_body for r in a)
    body = a[0].payload("m")
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["temperature"] == 0.0
    compact = build_workload("extract-compact", 5, seed=0)
    assert all(
        len(r.messages) == 2 and r.messages[0]["content"] == prompts.COMPACT_SYSTEM_PROMPT for r in compact
    )
    assert shared_prefix_chars(compact) == len(prompts.COMPACT_SYSTEM_PROMPT)
    assert set(WORKLOADS) >= {"extract-full", "extract-compact", "synthetic-long"}
    with pytest.raises(ValueError, match="unknown workload"):
        build_workload("nope", 1)


def test_synthetic_long_lengths_and_ignore_eos() -> None:
    reqs = build_workload("synthetic-long", 400, seed=0, lengths=OutputLengths(median=256, cap=1024))
    lens = [r.max_tokens for r in reqs]
    assert all(16 <= n <= 1024 for n in lens)
    assert 200 <= statistics.median(lens) <= 320
    assert all(r.extra_body == {"ignore_eos": True, "min_tokens": r.max_tokens} for r in reqs)
    assert reqs[0].payload("m")["ignore_eos"] is True
    assert lens == [r.max_tokens for r in build_workload("synthetic-long", 400, seed=0)]
    capped = OutputLengths(median=900, sigma=1.5, cap=1000)
    rng = random.Random(0)
    assert max(capped.sample(rng) for _ in range(500)) == 1000


def test_sampling_without_replacement_and_prompt_stats() -> None:
    picked = sample_without_replacement(list(range(10)), 25, random.Random(0))
    assert sorted(picked[:10]) == list(range(10)) and sorted(picked[10:20]) == list(range(10))
    reqs = build_workload("extract-compact", 40, postings=POSTINGS)
    assert len({r.messages[-1]["content"] for r in reqs}) == 40  # no repeats while the pool lasts
    stats = prompt_stats(reqs, count_tokens=lambda msgs: sum(len(m["content"]) for m in msgs) // 4)
    assert stats["n"] == 40
    assert stats["prompt_chars"]["min"] <= stats["prompt_chars"]["p50"] <= stats["prompt_chars"]["max"]
    assert stats["prompt_tokens"]["p50"] == pytest.approx(stats["prompt_chars"]["p50"] / 4, abs=1)


# ---------------------------------------------------------------------------------------------- loadgen


def _cfg(
    mode: str = "closed", warmup: int = 0, concurrency: int | None = None, rate: float | None = None
) -> RunConfig:
    return RunConfig("t", "t", "fake-model", "extract-compact", mode, concurrency, rate, warmup)


async def _with_server(cfg: FakeConfig, fn):
    server = FakeServer(cfg)
    await server.start()
    try:
        async with StreamingClient(server.base_url, "fake-model", max_connections=64) as client:
            return server, await fn(client)
    finally:
        await server.close()


def test_closed_loop_timings_match_configured_delays() -> None:
    cfg = FakeConfig(ttft_s=0.06, itl_s=0.01, output_tokens=12)

    async def go(client: StreamingClient) -> RunResult:
        return await run_closed_loop(client, _reqs(10, max_tokens=64), 4, _cfg(warmup=2, concurrency=4))

    server, run = asyncio.run(_with_server(cfg, go))
    assert server.requests_seen == 10  # warm-up requests were sent ...
    assert len(run.results) == 8  # ... but are not in the results
    for r in run.results:
        assert r.ok and r.status == 200
        assert r.completion_tokens == 12 and r.usage_source == "usage" and r.n_chunks == 12
        assert r.prompt_tokens and r.prompt_tokens > 0
        assert r.ttft_s is not None and 0.055 <= r.ttft_s <= 0.2
        assert r.itl_mean_s is not None and 0.008 <= r.itl_mean_s <= 0.03
        assert r.itl_max_s is not None and r.itl_max_s >= r.itl_p95_s >= r.itl_p50_s
        assert len(r.gaps_s) == 11
        assert 0.06 + 11 * 0.01 - 0.005 <= r.e2e_s <= 0.5
    assert 2 * 0.17 * 0.9 <= run.duration_s <= 2.0  # 8 requests / 4 workers = 2 rounds of ~0.17 s


def test_usage_fallback_counts_chunks_and_ignore_eos() -> None:
    cfg = FakeConfig(ttft_s=0.0, itl_s=0.0, output_tokens=5, send_usage=False)
    reqs = [BenchRequest("a", [{"role": "user", "content": "hi"}], 9, {"ignore_eos": True})]

    async def go(client: StreamingClient) -> RunResult:
        return await run_closed_loop(client, reqs, 1, _cfg())

    _, run = asyncio.run(_with_server(cfg, go))
    (r,) = run.results
    assert r.usage_source == "chunks" and r.completion_tokens == 9 and r.prompt_tokens is None


def test_errors_are_recorded_and_counted() -> None:
    cfg = FakeConfig(ttft_s=0.0, itl_s=0.001, output_tokens=3, fail_every=3)

    async def go(client: StreamingClient) -> RunResult:
        return await run_closed_loop(client, _reqs(9), 1, _cfg(concurrency=1))

    _, run = asyncio.run(_with_server(cfg, go))
    bad = [r for r in run.results if not r.ok]
    assert len(bad) == 3 and all(r.status == 500 and "HTTP 500" in r.error for r in bad)
    (s,) = summarize_rows(run.rows())
    assert s["errors"] == 3 and s["error_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["n"] == 9


def test_connection_error_is_an_error_row() -> None:
    async def go() -> RunResult:
        async with StreamingClient("http://127.0.0.1:9/v1", "m", timeout_s=2) as client:
            return await run_closed_loop(client, _reqs(2), 1, _cfg())

    run = asyncio.run(go())
    assert all(r.status == 0 and r.error and r.ttft_s is None for r in run.results)


def test_closed_loop_duration_mode_cycles_requests() -> None:
    cfg = FakeConfig(ttft_s=0.01, itl_s=0.005, output_tokens=4)

    async def go(client: StreamingClient) -> RunResult:
        return await run_closed_loop(client, _reqs(3), 2, _cfg(concurrency=2), duration_s=0.4)

    _, run = asyncio.run(_with_server(cfg, go))
    assert len(run.results) > 3  # cycled past the 3 distinct requests
    assert 0.35 <= run.duration_s <= 1.0
    assert max(r.start_s for r in run.results) < 0.4


def test_open_loop_poisson_arrivals() -> None:
    cfg = FakeConfig(ttft_s=0.02, itl_s=0.005, output_tokens=4)
    rate = 40.0

    async def go(client: StreamingClient) -> RunResult:
        return await run_open_loop(client, _reqs(31), rate, _cfg("poisson", warmup=1, rate=rate), seed=1)

    _, run = asyncio.run(_with_server(cfg, go))
    assert len(run.results) == 30 and all(r.ok for r in run.results)
    starts = sorted(r.start_s for r in run.results)
    span = starts[-1] - starts[0]
    # 29 exponential gaps at 40/s: mean span 0.725 s; generous bounds keep this robust on slow runners
    assert 0.25 <= span <= 2.0
    # open loop: arrivals overlap, unlike a single closed-loop worker
    assert max(r.end_s for r in run.results) < sum(r.e2e_s for r in run.results)


def test_open_loop_cap_counts_client_queueing() -> None:
    cfg = FakeConfig(ttft_s=0.05, itl_s=0.0, output_tokens=1)

    async def go(client: StreamingClient) -> RunResult:
        return await run_open_loop(client, _reqs(6), 1000.0, _cfg("poisson", rate=1000.0), max_concurrency=1)

    _, run = asyncio.run(_with_server(cfg, go))
    ttfts = sorted(r.ttft_s for r in run.results if r.ttft_s is not None)
    assert ttfts[-1] >= 5 * 0.05 * 0.9  # the last arrival waited behind five others


def test_sweep_levels_queueing_and_prefix_cache() -> None:
    cfg = FakeConfig(ttft_s=0.04, itl_s=0.002, output_tokens=4, max_num_seqs=2, prefix_cache_factor=0.25)
    reqs = build_workload("extract-full", 3 * 8, postings=POSTINGS, shots=[], max_tokens=8)

    async def go(client: StreamingClient) -> list[RunResult]:
        return await sweep(
            client,
            reqs,
            label="s",
            workload="extract-full",
            concurrencies=[1, 8],
            rates=[50.0],
            per_level=7,
            warmup=1,
        )

    server, runs = asyncio.run(_with_server(cfg, go))
    assert [r.config.mode for r in runs] == ["closed", "closed", "poisson"]
    assert [r.config.concurrency for r in runs[:2]] == [1, 8] and runs[2].config.rate == 50.0
    assert all(len(r.results) == 7 for r in runs)
    assert len({r.config.run_id for r in runs}) == 3
    assert server.prefix_hits == 3 * 8 - 1  # every request after the first reuses the system prompt
    s1, s8, _ = summarize_rows([row for r in runs for row in r.rows()])
    assert s1["ttft_p50_ms"] < 0.04 * 1000  # cached prefix: 0.25 x 40 ms
    assert s8["ttft_p95_ms"] > s1["ttft_p95_ms"] * 2  # 8 in flight, server runs 2 at a time -> queueing
    with pytest.raises(ValueError, match="need"):
        asyncio.run(
            _with_server(
                cfg, lambda c: sweep(c, reqs[:5], label="x", workload="w", concurrencies=[1, 2], per_level=4)
            )
        )


# ---------------------------------------------------------------------------------------------- metrics / cost


def _row(i: int, ttft: float, e2e: float, toks: int = 101, status: int = 200, **kw: object) -> dict:
    base = {
        "run_id": "r",
        "label": "L",
        "model": "m",
        "workload": "w",
        "mode": "closed",
        "concurrency": "4",
        "rate": "",
        "run_duration_s": "10",
        "request_id": str(i),
        "start_s": "0",
        "end_s": str(e2e),
        "ttft_s": str(ttft),
        "e2e_s": str(e2e),
        "itl_mean_s": "0.01",
        "itl_p95_s": "0.02",
        "prompt_tokens": "1000",
        "completion_tokens": str(toks),
        "status": str(status),
        "error": "" if status == 200 else "HTTP 500",
    }
    base.update({k: str(v) for k, v in kw.items()})
    return base


def test_summary_maths_goodput_and_cost() -> None:
    rows = [_row(i, ttft=0.1 * (i + 1), e2e=1.0 + i) for i in range(10)]  # ttft 0.1..1.0, e2e 1..10
    rows.append(_row(99, ttft=0, e2e=0, toks=0, status=500))
    (s,) = summarize_rows(rows, sla=SLA(ttft_s=0.5, e2e_s=4.0), gpu_usd_per_hour=0.80)
    assert s["n"] == 11 and s["errors"] == 1
    assert s["requests_per_s"] == 1.0  # 10 ok in a 10 s window
    assert s["output_tokens_per_s"] == pytest.approx(101.0)
    assert s["total_tokens_per_s"] == pytest.approx(1101.0)
    assert s["ttft_p50_ms"] == pytest.approx(550.0)
    assert s["e2e_p95_ms"] == pytest.approx(9550.0)
    assert s["itl_p50_ms"] == pytest.approx(10.0) and s["itl_worst_p95_ms"] == pytest.approx(20.0)
    # tpot of request 0: (1.0 - 0.1) / 100
    assert s["tpot_p50_ms"] > 0
    # SLA met by i = 0..3 (ttft <= 0.5 and e2e <= 4) -> 4 requests in 10 s
    assert s["goodput_rps"] == pytest.approx(0.4)
    assert s["sla_ok"] is False
    assert s["usd_per_1k_requests"] == pytest.approx(0.80 / 3600 * 1000, rel=1e-3)
    assert s["usd_per_1m_output_tokens"] == pytest.approx(0.80 / (101 * 3600) * 1e6, rel=1e-3)
    assert "| L | closed | 4 |" in markdown_table([s])


def test_cost_functions() -> None:
    assert usd_per_1k_requests(0.80, 1.0) == pytest.approx(0.2222, abs=1e-4)
    assert usd_per_1k_requests(0.80, 10.0) == pytest.approx(0.02222, abs=1e-5)
    assert usd_per_1m_output_tokens(0.80, 2000.0) == pytest.approx(0.1111, abs=1e-4)
    assert usd_per_1k_requests(0.80, 0.0) is None and usd_per_1m_output_tokens(1.0, 0) is None


def test_matrix_describes_plan_and_prices() -> None:
    m = yaml.safe_load(MATRIX.read_text())
    models = {c["model"] for c in m["configs"]}
    assert {
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
    } <= models
    seqs = {
        c.get("max_num_seqs", m["defaults"]["max_num_seqs"]) for c in m["configs"] if c.get("engine") is None
    }
    assert {16, 64, 256} <= seqs
    assert {c.get("enable_prefix_caching", True) for c in m["configs"]} == {True, False}
    assert any(c.get("quantization") == "fp8" for c in m["configs"])
    assert any(c.get("enable_lora") for c in m["configs"]) and any(
        c.get("engine") == "llama.cpp" for c in m["configs"]
    )
    assert all(w in WORKLOADS for c in m["configs"] for w in c.get("workloads", m["load"]["workloads"]))
    for name, p in m["gpu_prices"].items():
        assert {"usd_per_hour", "source", "checked"} <= set(p), name
    assert gpu_price(MATRIX, "L4") == pytest.approx(0.80)
    with pytest.raises(KeyError):
        gpu_price(MATRIX, "B300")


# ---------------------------------------------------------------------------------------------- gpu


def test_gpu_sampler_custom_query_and_noop() -> None:
    samples = iter([(100.0, 24000.0, 10.0), (300.0, 24000.0, 50.0), None])
    sampler = GpuSampler(interval_s=0.001, query=lambda: next(samples, None), backend="fake")
    with sampler:
        while len(sampler.samples) < 2:
            pass
    st = sampler.stats()
    assert st["backend"] == "fake" and st["mem_peak_mib"] == 300.0 and st["util_mean"] == 30.0
    off = GpuSampler(enabled=False)
    with off:
        pass
    assert off.stats() == {"backend": "none", "n_samples": 0}
    assert GpuSampler().stop()["backend"] in ("none", "nvml", "nvidia-smi")  # auto-detect never raises


def test_parse_nvidia_smi() -> None:
    assert parse_nvidia_smi("1024, 23034, 87\n") == (1024.0, 23034.0, 87.0)
    assert parse_nvidia_smi("100, 1000, 10\n300, 1000, 30\n") == (400.0, 2000.0, 20.0)
    assert parse_nvidia_smi("No devices were found") is None


# ---------------------------------------------------------------------------------------------- CLI


def test_cli_run_and_summarize_against_fake_server(tmp_path: Path) -> None:
    runner = CliRunner()
    raw = tmp_path / "raw" / "fake.csv"
    with running_fake_server(FakeConfig(ttft_s=0.01, itl_s=0.002, output_tokens=6)) as server:
        res = runner.invoke(
            app,
            [
                "bench", "run", "--base-url", server.base_url, "--model", "fake-model",
                "--workload", "extract-compact", "--concurrency", "1,4", "--rate", "100",
                "--num-requests", "8", "--warmup", "1", "--no-gpu", "--out", str(raw),
            ],
        )  # fmt: skip
    assert res.exit_code == 0, res.output
    with open(raw, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_FIELDS
        rows = list(reader)
    assert len(rows) == 3 * 8 and {r["mode"] for r in rows} == {"closed", "poisson"}
    assert all(r["status"] == "200" and r["completion_tokens"] == "6" for r in rows)

    out = tmp_path / "summary.csv"
    md = tmp_path / "summary.md"
    res = runner.invoke(
        app, ["bench", "summarize", str(raw), "--gpu", "L4", "--out", str(out), "--markdown", str(md)]
    )
    assert res.exit_code == 0, res.output
    summ = read_rows([out])
    assert len(summ) == 3
    for s in summ:
        assert float(s["gpu_usd_per_hour"]) == 0.80
        rps = float(s["requests_per_s"])
        assert math.isclose(float(s["usd_per_1k_requests"]), 0.80 / (rps * 3600) * 1000, rel_tol=1e-2)
        assert float(s["ttft_p50_ms"]) >= 9.0
    assert md.read_text().startswith("| label |")

    res = runner.invoke(app, ["bench", "summarize", str(raw), "--gpu-price", "2.0", "--out", str(out)])
    assert res.exit_code == 0 and all(float(s["gpu_usd_per_hour"]) == 2.0 for s in read_rows([out]))


def test_write_csv_round_trip(tmp_path: Path) -> None:
    cfg = FakeConfig(ttft_s=0.0, itl_s=0.0, output_tokens=2)

    async def go(client: StreamingClient) -> RunResult:
        return await run_closed_loop(client, _reqs(3), 1, _cfg(concurrency=1))

    _, run = asyncio.run(_with_server(cfg, go))
    p = write_csv(tmp_path / "x.csv", [run])
    rows = read_rows([p])
    assert len(rows) == 3 and rows[0]["gpu_backend"] == "none" and rows[0]["rate"] == ""
    assert "gaps_s" not in rows[0]


def test_cli_workload_stats() -> None:
    res = CliRunner().invoke(app, ["bench", "workload", "--workload", "extract-full", "--num-requests", "20"])
    assert res.exit_code == 0, res.output
    assert '"shared_prefix_chars"' in res.output and '"prompt_chars"' in res.output
