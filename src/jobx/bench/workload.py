"""Benchmark requests built from the repo's own postings.

Workloads (all greedy, ``temperature=0``):

* ``extract-full``: ``SYSTEM_PROMPT`` + the 3 few-shot exemplars as a shared prefix, then the posting. This is
  the base-model request used by the baselines, ~1.8K input tokens, and prefix-cache friendly because only
  the last user turn differs between requests.
* ``extract-compact``: ``COMPACT_SYSTEM_PROMPT`` and the posting, no shots (what the fine-tune was trained on).
* ``synthetic-long`` / ``synthetic-long-compact``: the same prompts, but each request's output length is
  drawn from a seeded lognormal and enforced with ``max_tokens`` plus vLLM's ``ignore_eos`` (and
  ``min_tokens``), to stress the decode phase independently of what the model would write.

Postings are sampled without replacement (reshuffled passes once the pool is exhausted), so a sweep over
several concurrency levels sends distinct postings as long as the pool is large enough; otherwise prefix
caching would also hit the posting itself and flatter the later levels.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from jobx import REPO_ROOT, prompts
from jobx.data.build import read_jsonl

DEFAULT_MAX_TOKENS = 384
DEFAULT_SPLITS: tuple[str, ...] = ("test_synth", "val")
SPLIT_PATHS = {
    "train": "data/processed/train.jsonl",
    "val": "data/processed/val.jsonl",
    "test_synth": "data/processed/test_synth.jsonl",
}


@dataclass(frozen=True)
class WorkloadSpec:
    system: str  # 'full' or 'compact' (prompts.SYSTEM_PROMPTS)
    shots: int
    synthetic_output: bool


WORKLOADS: dict[str, WorkloadSpec] = {
    "extract-full": WorkloadSpec("full", 3, False),
    "extract-compact": WorkloadSpec("compact", 0, False),
    "synthetic-long": WorkloadSpec("full", 3, True),
    "synthetic-long-compact": WorkloadSpec("compact", 0, True),
}


@dataclass(frozen=True)
class OutputLengths:
    """Seeded lognormal output lengths for the synthetic workloads."""

    median: int = 256
    sigma: float = 0.6
    minimum: int = 16
    cap: int = 1024

    def sample(self, rng: random.Random) -> int:
        n = round(rng.lognormvariate(math.log(self.median), self.sigma))
        return max(self.minimum, min(self.cap, n))


@dataclass
class BenchRequest:
    id: str
    messages: list[dict[str, str]]
    max_tokens: int
    extra_body: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_chars(self) -> int:
        return sum(len(m["content"]) for m in self.messages)

    def payload(self, model: str) -> dict[str, Any]:
        """Body of a streaming ``/v1/chat/completions`` request."""
        return {
            "model": model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            **self.extra_body,
        }


def load_postings(
    splits: Sequence[str] = DEFAULT_SPLITS, root: str | Path = REPO_ROOT
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in splits:
        rows.extend(read_jsonl(Path(root) / SPLIT_PATHS[s]))
    return rows


def sample_without_replacement(pool: Sequence[Any], n: int, rng: random.Random) -> list[Any]:
    """``n`` items; each pass over the pool is a fresh shuffle, so items repeat only after the pool is used up."""
    if not pool:
        raise ValueError("empty pool")
    out: list[Any] = []
    while len(out) < n:
        order = list(pool)
        rng.shuffle(order)
        out.extend(order[: n - len(out)])
    return out


def build_workload(
    name: str,
    n: int,
    *,
    seed: int = 0,
    postings: Sequence[dict[str, Any]] | None = None,
    shots: Sequence[dict[str, Any]] | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    lengths: OutputLengths | None = None,
    root: str | Path = REPO_ROOT,
) -> list[BenchRequest]:
    """``n`` deterministic requests for workload ``name`` (same seed -> identical requests)."""
    if name not in WORKLOADS:
        raise ValueError(f"unknown workload {name!r}; choose from {sorted(WORKLOADS)}")
    spec = WORKLOADS[name]
    rng = random.Random(seed)
    pool = list(postings) if postings is not None else load_postings(root=root)
    if spec.shots and shots is None:
        shots = prompts.select_few_shot(read_jsonl(Path(root) / SPLIT_PATHS["train"]), k=spec.shots)
    shot_list = list(shots or [])[: spec.shots]
    shot_ids = {s["id"] for s in shot_list}
    pool = [p for p in pool if p["id"] not in shot_ids]
    lengths = lengths or OutputLengths()
    out: list[BenchRequest] = []
    for i, post in enumerate(sample_without_replacement(pool, n, rng)):
        msgs = prompts.messages(post["text"], shot_list, system=spec.system)
        if spec.synthetic_output:
            k = lengths.sample(rng)
            out.append(BenchRequest(f"{i:05d}-{post['id']}", msgs, k, {"ignore_eos": True, "min_tokens": k}))
        else:
            out.append(BenchRequest(f"{i:05d}-{post['id']}", msgs, max_tokens))
    return out


def shared_prefix_chars(requests: Sequence[BenchRequest]) -> int:
    """Characters of the leading messages that every request shares (what prefix caching can reuse)."""
    if not requests:
        return 0
    first = requests[0].messages
    k = 0
    while k < len(first) and all(len(r.messages) > k and r.messages[k] == first[k] for r in requests):
        k += 1
    return sum(len(m["content"]) for m in first[:k])


def _dist(values: Sequence[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {
        "min": float(a.min()),
        "p50": float(np.percentile(a, 50)),
        "mean": round(float(a.mean()), 1),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def prompt_stats(
    requests: Sequence[BenchRequest], count_tokens: Callable[[list[dict[str, str]]], int] | None = None
) -> dict[str, Any]:
    """Prompt-length distribution in characters, and in tokens when ``count_tokens`` is given."""
    stats: dict[str, Any] = {
        "n": len(requests),
        "prompt_chars": _dist([r.prompt_chars for r in requests]),
        "shared_prefix_chars": shared_prefix_chars(requests),
        "max_tokens": _dist([r.max_tokens for r in requests]),
    }
    if count_tokens is not None:
        stats["prompt_tokens"] = _dist([count_tokens(r.messages) for r in requests])
    return stats


def hf_token_counter(tokenizer_id: str) -> Callable[[list[dict[str, str]]], int]:
    """Token counter from a Hugging Face chat template (optional: needs ``transformers``)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_id)

    def count(messages: list[dict[str, str]]) -> int:
        return len(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    return count
