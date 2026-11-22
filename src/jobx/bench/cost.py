"""Serving cost from a GPU's hourly price and measured throughput.

Assumes the GPU is fully busy at the measured rate (a sustained-load figure, not including idle time,
cold starts or the minimum billing increment), so it is a lower bound for bursty traffic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def usd_per_1k_requests(gpu_usd_per_hour: float, requests_per_s: float) -> float | None:
    """``gpu_usd_per_hour / (requests_per_s * 3600) * 1000``; ``None`` when nothing completed."""
    if requests_per_s <= 0:
        return None
    return gpu_usd_per_hour / (requests_per_s * 3600.0) * 1000.0


def usd_per_1m_output_tokens(gpu_usd_per_hour: float, output_tokens_per_s: float) -> float | None:
    if output_tokens_per_s <= 0:
        return None
    return gpu_usd_per_hour / (output_tokens_per_s * 3600.0) * 1_000_000.0


def gpu_prices(matrix_path: str | Path) -> dict[str, dict[str, Any]]:
    """The ``gpu_prices`` table of ``configs/bench/matrix.yaml`` keyed by GPU name."""
    data = yaml.safe_load(Path(matrix_path).read_text())
    return {str(k): dict(v) for k, v in (data.get("gpu_prices") or {}).items()}


def gpu_price(matrix_path: str | Path, gpu: str) -> float:
    prices = gpu_prices(matrix_path)
    if gpu not in prices:
        raise KeyError(f"no price for GPU {gpu!r} in {matrix_path}; known: {sorted(prices)}")
    return float(prices[gpu]["usd_per_hour"])
