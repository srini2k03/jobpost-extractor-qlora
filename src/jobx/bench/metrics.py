"""Aggregate raw per-request CSV rows into one summary row per run.

Definitions (successful requests only, except the error rate):

* ``ttft_*``, ``e2e_*``: percentiles across requests.
* ``tpot_*``: time per output token after the first, ``(e2e - ttft) / (completion_tokens - 1)``.
* ``itl_*``: percentiles across requests of each request's mean inter-token gap; ``itl_worst_p95_ms`` is the
  p95 across requests of each request's own p95 gap (decode stalls, e.g. when a long prefill is scheduled).
* Throughput over the run's measurement window (``run_duration_s``, warm-up excluded): requests/s, output
  tokens/s and total (prompt + output) tokens/s.
* Goodput: requests/s that completed with TTFT <= ``ttft_s`` and E2E <= ``e2e_s`` (a per-request SLA);
  ``sla_ok`` additionally says whether the run's p95 TTFT and p95 E2E are within the thresholds.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jobx.bench.cost import usd_per_1k_requests, usd_per_1m_output_tokens

PCTS = (50, 95, 99)
GROUP_KEYS = ("run_id", "label", "model", "workload", "mode", "concurrency", "rate")


@dataclass(frozen=True)
class SLA:
    ttft_s: float = 1.0
    e2e_s: float = 5.0


def read_rows(paths: Iterable[str | Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def _f(v: Any) -> float | None:
    if v is None or v == "":
        return None
    return float(v)


def _pcts(values: Sequence[float], name: str, scale: float = 1000.0) -> dict[str, float | None]:
    a = np.asarray(values, dtype=float)
    return {
        f"{name}_p{q}_ms": (round(float(np.percentile(a, q)) * scale, 2) if a.size else None) for q in PCTS
    }


def summarize_run(rows: Sequence[dict[str, Any]], sla: SLA = SLA()) -> dict[str, Any]:
    """One summary dict for the rows of a single run."""
    first = rows[0]
    ok = [r for r in rows if str(r["status"]) == "200" and not r.get("error")]
    n, n_ok = len(rows), len(ok)
    duration = _f(first.get("run_duration_s"))
    if not duration:
        duration = max(float(r["end_s"]) for r in rows) - min(float(r["start_s"]) for r in rows)
    ttft = [float(r["ttft_s"]) for r in ok if r.get("ttft_s") not in (None, "")]
    e2e = [float(r["e2e_s"]) for r in ok]
    itl = [float(r["itl_mean_s"]) for r in ok if r.get("itl_mean_s") not in (None, "")]
    itl95 = [float(r["itl_p95_s"]) for r in ok if r.get("itl_p95_s") not in (None, "")]
    tpot = [
        (float(r["e2e_s"]) - float(r["ttft_s"])) / (int(r["completion_tokens"]) - 1)
        for r in ok
        if r.get("ttft_s") not in (None, "") and int(r["completion_tokens"]) > 1
    ]
    out_tok = sum(int(r["completion_tokens"]) for r in ok)
    in_tok = sum(int(r["prompt_tokens"]) for r in ok if r.get("prompt_tokens") not in (None, ""))
    good = sum(
        1
        for r in ok
        if r.get("ttft_s") not in (None, "")
        and float(r["ttft_s"]) <= sla.ttft_s
        and float(r["e2e_s"]) <= sla.e2e_s
    )
    rps = n_ok / duration if duration > 0 else 0.0
    out: dict[str, Any] = {k: first.get(k, "") for k in GROUP_KEYS}
    out.update(
        {
            "n": n,
            "errors": n - n_ok,
            "error_rate": round((n - n_ok) / n, 4) if n else None,
            "duration_s": round(duration, 3),
            "requests_per_s": round(rps, 3),
            "output_tokens_per_s": round(out_tok / duration, 1) if duration > 0 else 0.0,
            "total_tokens_per_s": round((out_tok + in_tok) / duration, 1) if duration > 0 else 0.0,
            "mean_prompt_tokens": round(in_tok / n_ok, 1) if n_ok and in_tok else None,
            "mean_output_tokens": round(out_tok / n_ok, 1) if n_ok else None,
        }
    )
    out.update(_pcts(ttft, "ttft"))
    out.update(_pcts(itl, "itl"))
    out["itl_worst_p95_ms"] = round(float(np.percentile(itl95, 95)) * 1000, 2) if itl95 else None
    out.update(_pcts(tpot, "tpot"))
    out.update(_pcts(e2e, "e2e"))
    out["goodput_rps"] = round(good / duration, 3) if duration > 0 else 0.0
    out["sla_ttft_s"] = sla.ttft_s
    out["sla_e2e_s"] = sla.e2e_s
    p95_ttft, p95_e2e = out["ttft_p95_ms"], out["e2e_p95_ms"]
    out["sla_ok"] = bool(
        n_ok
        and p95_ttft is not None
        and p95_ttft <= sla.ttft_s * 1000
        and p95_e2e is not None
        and p95_e2e <= sla.e2e_s * 1000
    )
    out["gpu_mem_peak_mib"] = _f(first.get("gpu_mem_peak_mib"))
    out["gpu_util_mean"] = _f(first.get("gpu_util_mean"))
    return out


def add_cost(summary: dict[str, Any], gpu_usd_per_hour: float | None) -> dict[str, Any]:
    if gpu_usd_per_hour is None:
        summary.update(gpu_usd_per_hour=None, usd_per_1k_requests=None, usd_per_1m_output_tokens=None)
        return summary
    c1k = usd_per_1k_requests(gpu_usd_per_hour, summary["requests_per_s"])
    c1m = usd_per_1m_output_tokens(gpu_usd_per_hour, summary["output_tokens_per_s"])
    summary["gpu_usd_per_hour"] = gpu_usd_per_hour
    summary["usd_per_1k_requests"] = round(c1k, 5) if c1k is not None else None
    summary["usd_per_1m_output_tokens"] = round(c1m, 4) if c1m is not None else None
    return summary


def summarize_rows(
    rows: Sequence[dict[str, Any]], *, sla: SLA = SLA(), gpu_usd_per_hour: float | None = None
) -> list[dict[str, Any]]:
    """Group raw rows by run and summarise each group (order of first appearance)."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k, "") for k in GROUP_KEYS), []).append(r)
    return [add_cost(summarize_run(g, sla), gpu_usd_per_hour) for g in groups.values()]


def write_summary(path: str | Path, summaries: Sequence[dict[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for s in summaries:
        keys.extend(k for k in s if k not in keys)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for s in summaries:
            w.writerow({k: "" if s.get(k) is None else s.get(k) for k in keys})
    return p


TABLE_COLUMNS = (
    ("label", "label"),
    ("mode", "mode"),
    ("concurrency", "conc"),
    ("rate", "rate"),
    ("n", "n"),
    ("error_rate", "err"),
    ("requests_per_s", "req/s"),
    ("output_tokens_per_s", "out tok/s"),
    ("ttft_p50_ms", "TTFT p50"),
    ("ttft_p95_ms", "TTFT p95"),
    ("itl_p50_ms", "ITL p50"),
    ("e2e_p50_ms", "E2E p50"),
    ("e2e_p95_ms", "E2E p95"),
    ("e2e_p99_ms", "E2E p99"),
    ("goodput_rps", "goodput"),
    ("usd_per_1k_requests", "$/1K req"),
)


def markdown_table(summaries: Sequence[dict[str, Any]]) -> str:
    """Compact Markdown table (latencies in ms)."""
    head = "| " + " | ".join(h for _, h in TABLE_COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in TABLE_COLUMNS) + "|"
    lines = [head, sep]
    for s in summaries:
        cells = []
        for k, _ in TABLE_COLUMNS:
            v = s.get(k)
            cells.append(
                "-" if v is None or v == "" else (f"{v:.4f}" if k == "usd_per_1k_requests" else str(v))
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
