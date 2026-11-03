"""Planning for CPU training: sequence length from measured token lengths, and how much data fits the
time budget (from a measured seconds-per-step benchmark). Pure functions, unit-tested without torch."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def length_stats(lengths: Sequence[int]) -> dict[str, float]:
    a = np.asarray(lengths, dtype=float)
    return {f"p{q}": float(np.percentile(a, q)) for q in (50, 90, 95, 99)} | {
        "max": float(a.max()),
        "n": float(len(a)),
    }


def choose_max_length(
    lengths: Sequence[int], *, target: int = 512, cap: int = 768, multiple: int = 64
) -> int:
    """`target` if the p99 fits, else p99 rounded up to `multiple`, never above `cap`. Examples longer
    than the result are dropped from training (never truncated: cutting the prompt would change the task,
    cutting the completion would teach broken JSON)."""
    p99 = float(np.percentile(np.asarray(lengths, dtype=float), 99))
    if p99 <= target:
        return target
    return int(min(cap, math.ceil(p99 / multiple) * multiple))


def _rank(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()


def stratified_subset(examples: Sequence[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Deterministic subset of size n, proportional over (template family, domain) strata."""
    if n >= len(examples):
        return list(examples)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for e in examples:
        groups[(e["meta"].get("family", "?"), e["meta"].get("domain", "?"))].append(e)
    total = len(examples)
    quota = {k: n * len(v) / total for k, v in groups.items()}
    take = {k: int(q) for k, q in quota.items()}
    rest = n - sum(take.values())
    for k in sorted(quota, key=lambda k: (-(quota[k] - take[k]), k))[:rest]:
        take[k] += 1
    out = []
    for k, v in groups.items():
        out += sorted(v, key=lambda e: _rank(e["id"]))[: take[k]]
    return sorted(out, key=lambda e: e["id"])


@dataclass
class TrainPlan:
    seconds_per_step: float
    seconds_per_example: float
    effective_batch: int
    n_train_available: int
    n_train: int
    epochs: int
    max_steps: int
    est_hours: float
    jobs_needed: int
    max_length: int
    dropped_too_long: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_training(
    *,
    seconds_per_step: float,
    effective_batch: int,
    n_available: int,
    max_length: int,
    dropped_too_long: int = 0,
    job_minutes: float = 300,
    max_jobs: int = 3,
    overhead: float = 0.85,
    max_epochs: int = 2,
) -> TrainPlan:
    """Largest of {max_epochs .. 1} epochs over all data that fits the budget, else one epoch over a
    stratified subset that fits (rounded down to a whole number of optimiser steps)."""
    spe = seconds_per_step / effective_batch
    budget = job_minutes * 60 * max_jobs * overhead
    per_job = job_minutes * 60 * overhead
    for ep in range(max_epochs, 0, -1):
        steps = math.ceil(n_available * ep / effective_batch)
        if steps * seconds_per_step <= budget:
            n, epochs, reason = (
                n_available,
                ep,
                f"{ep} epoch(s) over all {n_available} training examples fit the budget",
            )
            break
    else:
        n = int(budget / spe) // effective_batch * effective_batch
        n = max(effective_batch, min(n, n_available))
        epochs, steps = 1, math.ceil(n / effective_batch)
        reason = (
            f"1 epoch over all data needs {n_available * spe / 3600:.1f} h > budget "
            f"{budget / 3600:.1f} h: stratified subset of {n}"
        )
    est = steps * seconds_per_step
    return TrainPlan(
        round(seconds_per_step, 2),
        round(spe, 3),
        effective_batch,
        n_available,
        n,
        epochs,
        steps,
        round(est / 3600, 2),
        max(1, math.ceil(est / per_job)),
        max_length,
        dropped_too_long,
        reason,
    )
