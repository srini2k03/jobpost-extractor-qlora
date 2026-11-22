"""Async streaming load generator for OpenAI-compatible ``/v1/chat/completions`` servers (vLLM, llama.cpp).

Every request is sent with ``stream=true`` and ``stream_options={"include_usage": true}``; each SSE line is
timestamped with ``time.perf_counter`` as it arrives. Per request we keep:

* TTFT: request start -> first chunk carrying content (text or reasoning);
* inter-token gaps between consecutive content chunks (mean/p50/p95/max per request; the list itself is kept
  in memory only, not written to the CSV);
* end-to-end latency, HTTP status, error text;
* prompt/completion tokens from the final usage chunk, falling back to the number of content chunks.

Two modes: closed loop (``concurrency`` workers, each sends its next request as soon as the previous one
finishes, for N requests or D seconds) and open loop (Poisson arrivals at ``rate`` requests/s, independent
of completions). Warm-up requests are sent first and excluded from the results.
"""

from __future__ import annotations

import asyncio
import csv
import gzip
import io
import json
import random
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import IO, Any, Protocol

import httpx
import numpy as np

from jobx.bench.gpu import GpuSampler
from jobx.bench.workload import BenchRequest


@dataclass
class RequestResult:
    request_id: str
    start_s: float  # relative to the run's t0
    end_s: float
    ttft_s: float | None
    e2e_s: float
    itl_mean_s: float | None
    itl_p50_s: float | None
    itl_p95_s: float | None
    itl_max_s: float | None
    n_chunks: int
    prompt_tokens: int | None
    completion_tokens: int
    usage_source: str  # 'usage' or 'chunks'
    status: int  # HTTP status, 0 when the request never got a response
    error: str
    prompt_chars: int
    max_tokens: int
    gaps_s: list[float] = field(default_factory=list, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == 200 and not self.error


@dataclass
class RunConfig:
    """Run-level settings, repeated on every CSV row so a raw CSV is self-describing."""

    run_id: str
    label: str
    model: str
    workload: str
    mode: str  # 'closed' or 'poisson'
    concurrency: int | None = None
    rate: float | None = None
    warmup: int = 0


class LevelMonitor(Protocol):
    """Something sampled while one level runs (GPU sampler, server-metrics scraper)."""

    def __enter__(self) -> Any: ...

    def __exit__(self, *exc: Any) -> Any: ...

    def stats(self) -> dict[str, Any]: ...


SERVER_FIELDS = ("kv_cache_usage_mean", "kv_cache_usage_max", "prefix_cache_hit_rate", "num_preemptions")


@dataclass
class RunResult:
    config: RunConfig
    results: list[RequestResult]
    duration_s: float
    gpu: dict[str, Any] = field(default_factory=dict)
    server: dict[str, Any] = field(default_factory=dict)

    def rows(self) -> list[dict[str, Any]]:
        base = {
            **asdict(self.config),
            "run_duration_s": round(self.duration_s, 6),
            "gpu_backend": self.gpu.get("backend", "none"),
            "gpu_mem_peak_mib": self.gpu.get("mem_peak_mib"),
            "gpu_mem_total_mib": self.gpu.get("mem_total_mib"),
            "gpu_util_mean": self.gpu.get("util_mean"),
            **{k: self.server.get(k) for k in SERVER_FIELDS},
        }
        out = []
        for r in self.results:
            d = {k: v for k, v in asdict(r).items() if k != "gaps_s"}
            out.append({**base, **d})
        return out


CSV_FIELDS: list[str] = [
    *(f.name for f in fields(RunConfig)),
    "run_duration_s",
    "gpu_backend",
    "gpu_mem_peak_mib",
    "gpu_mem_total_mib",
    "gpu_util_mean",
    *SERVER_FIELDS,
    *(f.name for f in fields(RequestResult) if f.name != "gaps_s"),
]


def open_csv(path: str | Path, mode: str = "r") -> IO[str]:
    """Text handle for a CSV; ``.gz`` paths are gzip-compressed with a fixed mtime (reproducible bytes)."""
    p = Path(path)
    if p.suffix == ".gz":
        raw = open(p, mode + "b")  # noqa: SIM115 - closed with the wrapper
        gz = gzip.GzipFile(fileobj=raw, mode=mode + "b", mtime=0)
        return _ClosingWrapper(gz, raw)
    return open(p, mode, newline="", encoding="utf-8")


class _ClosingWrapper(io.TextIOWrapper):
    def __init__(self, gz: gzip.GzipFile, raw: IO[bytes]) -> None:
        super().__init__(gz, encoding="utf-8", newline="")
        self._raw = raw

    def close(self) -> None:
        super().close()
        self._raw.close()


def write_csv(path: str | Path, runs: Sequence[RunResult]) -> Path:
    """One row per request; a ``.csv.gz`` path writes it gzip-compressed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open_csv(p, "w") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for run in runs:
            for row in run.rows():
                w.writerow({k: _fmt(v) for k, v in row.items()})
    return p


def _fmt(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float):
        return round(v, 6)
    return v


def _chunk_has_content(chunk: dict[str, Any]) -> bool:
    for ch in chunk.get("choices") or []:
        delta = ch.get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
            return True
    return False


class StreamingClient:
    """Thin async client around one ``httpx.AsyncClient`` (connection pool sized for the run)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 600.0,
        max_connections: int | None = None,
        trust_env: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        # trust_env=False by default: an HTTP(S)_PROXY from the environment would add a hop and distort timing.
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=30.0),
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
            trust_env=trust_env,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> StreamingClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def send(self, req: BenchRequest, t0: float) -> RequestResult:
        start = time.perf_counter()
        first: float | None = None
        last: float | None = None
        gaps: list[float] = []
        n_chunks = 0
        usage: dict[str, Any] | None = None
        status = 0
        error = ""
        try:
            async with self._client.stream("POST", self.url, json=req.payload(self.model)) as resp:
                status = resp.status_code
                if status != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    error = f"HTTP {status}: {body[:200]}"
                else:
                    async for line in resp.aiter_lines():
                        now = time.perf_counter()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        if chunk.get("error"):
                            error = f"stream error: {str(chunk['error'])[:200]}"
                            break
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        if _chunk_has_content(chunk):
                            n_chunks += 1
                            if first is None:
                                first = now
                            elif last is not None:
                                gaps.append(now - last)
                            last = now
        except (httpx.HTTPError, json.JSONDecodeError, OSError) as e:
            error = f"{type(e).__name__}: {e}"[:300]
        end = time.perf_counter()
        if not error and first is None:
            error = "no content in stream"
        g = np.asarray(gaps, dtype=float)
        has_gaps = g.size > 0
        if usage and usage.get("completion_tokens") is not None:
            completion, source = int(usage["completion_tokens"]), "usage"
        else:
            completion, source = n_chunks, "chunks"
        prompt_tokens = usage.get("prompt_tokens") if usage else None
        return RequestResult(
            request_id=req.id,
            start_s=start - t0,
            end_s=end - t0,
            ttft_s=(first - start) if first is not None else None,
            e2e_s=end - start,
            itl_mean_s=float(g.mean()) if has_gaps else None,
            itl_p50_s=float(np.percentile(g, 50)) if has_gaps else None,
            itl_p95_s=float(np.percentile(g, 95)) if has_gaps else None,
            itl_max_s=float(g.max()) if has_gaps else None,
            n_chunks=n_chunks,
            prompt_tokens=int(prompt_tokens) if prompt_tokens is not None else None,
            completion_tokens=completion,
            usage_source=source,
            status=status,
            error=error,
            prompt_chars=req.prompt_chars,
            max_tokens=req.max_tokens,
            gaps_s=gaps,
        )


async def _closed(
    client: StreamingClient,
    requests: Iterator[BenchRequest],
    concurrency: int,
    t0: float,
    deadline: float | None,
) -> list[RequestResult]:
    results: list[RequestResult] = []

    async def worker() -> None:
        while deadline is None or time.perf_counter() < deadline:
            req = next(requests, None)
            if req is None:
                return
            results.append(await client.send(req, t0))

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return results


async def _warmup(client: StreamingClient, requests: Sequence[BenchRequest], concurrency: int) -> None:
    if requests:
        await _closed(
            client, iter(requests), max(1, min(concurrency, len(requests))), time.perf_counter(), None
        )


def _split(requests: Sequence[BenchRequest], warmup: int) -> tuple[list[BenchRequest], list[BenchRequest]]:
    if warmup >= len(requests):
        raise ValueError(f"warmup ({warmup}) must be smaller than the number of requests ({len(requests)})")
    return list(requests[:warmup]), list(requests[warmup:])


async def run_closed_loop(
    client: StreamingClient,
    requests: Sequence[BenchRequest],
    concurrency: int,
    config: RunConfig,
    *,
    duration_s: float | None = None,
) -> RunResult:
    """``concurrency`` workers. The first ``config.warmup`` requests warm up and are dropped; then every remaining
    request is sent once, or (with ``duration_s``) until the deadline, whichever comes first (requests in flight
    at the deadline still complete and count). Size the pool for the duration; it also caps the CSV size."""
    warm, measured = _split(requests, config.warmup)
    await _warmup(client, warm, concurrency)
    it: Iterator[BenchRequest] = iter(measured)
    t0 = time.perf_counter()
    deadline = t0 + duration_s if duration_s else None
    results = await _closed(client, it, concurrency, t0, deadline)
    return RunResult(config, sorted(results, key=lambda r: r.start_s), time.perf_counter() - t0)


async def run_open_loop(
    client: StreamingClient,
    requests: Sequence[BenchRequest],
    rate: float,
    config: RunConfig,
    *,
    duration_s: float | None = None,
    seed: int = 0,
    max_concurrency: int | None = None,
) -> RunResult:
    """Poisson arrivals at ``rate`` requests/s (seeded exponential gaps), regardless of completions. Warm-up
    requests run closed-loop first. ``max_concurrency`` optionally caps requests in flight (late arrivals then
    queue client-side; their queueing time counts towards TTFT and E2E, as it would for a real user)."""
    if rate <= 0:
        raise ValueError("rate must be > 0")
    warm, measured = _split(requests, config.warmup)
    await _warmup(client, warm, max_concurrency or 8)
    rng = random.Random(seed)
    sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def one(req: BenchRequest) -> RequestResult:
        if sem is None:
            return await client.send(req, t0)
        queued = time.perf_counter()
        async with sem:
            r = await client.send(req, t0)
        wait = r.start_s - (queued - t0)
        r.start_s -= wait
        r.e2e_s += wait
        if r.ttft_s is not None:
            r.ttft_s += wait
        return r

    tasks: list[asyncio.Task[RequestResult]] = []
    it: Iterator[BenchRequest] = iter(measured)
    t0 = time.perf_counter()
    next_at = t0
    for req in it:
        if duration_s and next_at - t0 >= duration_s:
            break
        delay = next_at - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(one(req)))
        next_at += rng.expovariate(rate)
    results = list(await asyncio.gather(*tasks))
    return RunResult(config, sorted(results, key=lambda r: r.start_s), time.perf_counter() - t0)


async def sweep(
    client: StreamingClient,
    requests: Sequence[BenchRequest],
    *,
    label: str,
    workload: str,
    concurrencies: Sequence[int] = (),
    rates: Sequence[float] = (),
    per_level: int,
    warmup: int = 0,
    duration_s: float | None = None,
    seed: int = 0,
    max_concurrency: int | None = None,
    gpu_sampler_factory: Callable[[], GpuSampler] | None = None,
    server_monitor_factory: Callable[[], LevelMonitor] | None = None,
) -> list[RunResult]:
    """One run per concurrency level (closed loop) and per rate (open loop). ``requests`` is split into
    consecutive slices of ``warmup + per_level`` so levels do not resend the same postings while the pool lasts.
    ``gpu_sampler_factory`` builds a fresh :class:`~jobx.bench.gpu.GpuSampler` per level (its window includes the
    level's warm-up); ``server_monitor_factory`` likewise for a server-metrics scraper (vLLM ``/metrics``)."""
    levels: list[tuple[str, float]] = [("closed", c) for c in concurrencies] + [("poisson", r) for r in rates]
    size = warmup + per_level
    if len(requests) < size * len(levels):
        raise ValueError(f"need {size * len(levels)} requests for {len(levels)} levels, got {len(requests)}")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    runs: list[RunResult] = []
    for i, (mode, level) in enumerate(levels):
        chunk = requests[i * size : (i + 1) * size]
        tag = f"c{int(level)}" if mode == "closed" else f"r{level:g}"
        cfg = RunConfig(
            run_id=f"{label}-{tag}-{stamp}",
            label=label,
            model=client.model,
            workload=workload,
            mode=mode,
            concurrency=int(level) if mode == "closed" else max_concurrency,
            rate=float(level) if mode == "poisson" else None,
            warmup=warmup,
        )
        sampler = gpu_sampler_factory() if gpu_sampler_factory else GpuSampler(enabled=False)
        monitor = server_monitor_factory() if server_monitor_factory else None
        if monitor is not None:
            monitor.__enter__()
        try:
            with sampler:
                if mode == "closed":
                    run = await run_closed_loop(client, chunk, int(level), cfg, duration_s=duration_s)
                else:
                    run = await run_open_loop(
                        client,
                        chunk,
                        level,
                        cfg,
                        duration_s=duration_s,
                        seed=seed + i,
                        max_concurrency=max_concurrency,
                    )
        finally:
            if monitor is not None:
                monitor.__exit__(None, None, None)
        run.gpu = sampler.stats()
        run.server = monitor.stats() if monitor is not None else {}
        runs.append(run)
    return runs
