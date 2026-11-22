"""Scrape an inference server's Prometheus ``/metrics`` while a load level runs.

vLLM (V1 engine) exposes the KV-cache usage gauge and prefix-cache counters; names changed across releases,
so each quantity accepts several aliases (the first one present wins). llama.cpp's ``llama-server
--metrics`` exposes ``llamacpp:kv_cache_usage_ratio``. Missing metrics simply give ``None``.

Per level the scraper reports the mean and max KV-cache usage (0-1), the prefix-cache hit rate over the
level (delta hits / delta queries, in tokens) and the number of preemptions during the level.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from typing import Any

ALIASES: dict[str, tuple[str, ...]] = {
    "kv_cache_usage": (
        "vllm:kv_cache_usage_perc",
        "vllm:gpu_cache_usage_perc",
        "llamacpp:kv_cache_usage_ratio",
    ),
    "prefix_hits": (
        "vllm:prefix_cache_hits_total",
        "vllm:gpu_prefix_cache_hits_total",
        "vllm:prefix_cache_hits",
    ),
    "prefix_queries": (
        "vllm:prefix_cache_queries_total",
        "vllm:gpu_prefix_cache_queries_total",
        "vllm:prefix_cache_queries",
    ),
    "preemptions": ("vllm:num_preemptions_total", "vllm:num_preemptions"),
    "running": ("vllm:num_requests_running", "llamacpp:requests_processing"),
    "waiting": ("vllm:num_requests_waiting", "llamacpp:requests_deferred"),
}

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


def parse_prometheus(text: str) -> dict[str, float]:
    """Metric name -> value summed over label sets (``# HELP``/``# TYPE`` lines and NaNs skipped)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        if v != v:  # NaN
            continue
        out[m.group(1)] = out.get(m.group(1), 0.0) + v
    return out


def pick(metrics: dict[str, float], quantity: str) -> float | None:
    for name in ALIASES[quantity]:
        if name in metrics:
            return metrics[name]
    return None


def summarize_samples(samples: list[dict[str, float]]) -> dict[str, Any]:
    kv = [v for s in samples if (v := pick(s, "kv_cache_usage")) is not None]
    out: dict[str, Any] = {
        "n_scrapes": len(samples),
        "kv_cache_usage_mean": round(sum(kv) / len(kv), 4) if kv else None,
        "kv_cache_usage_max": round(max(kv), 4) if kv else None,
        "prefix_cache_hit_rate": None,
        "num_preemptions": None,
    }
    if len(samples) >= 2:
        first, last = samples[0], samples[-1]
        h0, h1 = pick(first, "prefix_hits"), pick(last, "prefix_hits")
        q0, q1 = pick(first, "prefix_queries"), pick(last, "prefix_queries")
        if h0 is not None and h1 is not None and q0 is not None and q1 is not None and q1 > q0:
            out["prefix_cache_hit_rate"] = round((h1 - h0) / (q1 - q0), 4)
        p0, p1 = pick(first, "preemptions"), pick(last, "preemptions")
        if p0 is not None and p1 is not None:
            out["num_preemptions"] = int(p1 - p0)
    running = [v for s in samples if (v := pick(s, "running")) is not None]
    waiting = [v for s in samples if (v := pick(s, "waiting")) is not None]
    out["running_max"] = max(running) if running else None
    out["waiting_max"] = max(waiting) if waiting else None
    return out


def http_fetch(url: str, timeout_s: float = 5.0) -> str | None:
    import httpx

    try:
        r = httpx.get(url, timeout=timeout_s, trust_env=False)
    except httpx.HTTPError:
        return None
    return r.text if r.status_code == 200 else None


class MetricsScraper:
    """Poll ``url`` every ``interval_s`` in a daemon thread; a first and a final scrape bracket the level."""

    def __init__(
        self, url: str, interval_s: float = 1.0, fetch: Callable[[str], str | None] = http_fetch
    ) -> None:
        self.url = url
        self.interval_s = interval_s
        self.fetch = fetch
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _scrape(self) -> None:
        text = self.fetch(self.url)
        if text:
            parsed = parse_prometheus(text)
            if parsed:
                self.samples.append(parsed)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._scrape()

    def __enter__(self) -> MetricsScraper:
        self._scrape()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="metrics-scraper", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        self._scrape()

    def stats(self) -> dict[str, Any]:
        return summarize_samples(self.samples)
