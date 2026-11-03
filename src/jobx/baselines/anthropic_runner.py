"""Optional hosted-API baseline on the Anthropic API (the default API row uses the free Gemini tier
through :mod:`jobx.baselines.openai_compat_runner`): the same prompt and schema, sent to an Anthropic model.

* zero-shot and few-shot runs use :func:`jobx.prompts.messages` exactly like the open-model baselines;
* structured outputs (``output_config.format`` with the ``JobPosting`` JSON Schema) constrain the reply
  to schema-shaped JSON;
* async requests with a concurrency limit, retries that honour ``retry-after`` on 429 and back off on
  5xx/overload/connection errors;
* per-request token and cost accounting, a cost estimate before anything runs (``count_tokens``),
  and a hard budget guard while running;
* output is the standard predictions JSONL + ``.meta.json``, scored by ``jobx eval`` like every row.

The API key is read by the SDK from ``ANTHROPIC_API_KEY`` and is never printed or written anywhere.

    python -m jobx.baselines.anthropic_runner --model claude-opus-5-5 --shots 0,3 --max-examples 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jobx import REPO_ROOT, prompts
from jobx.data.build import read_jsonl
from jobx.eval.run import SPLIT_FILES, eval_subset, git_commit, write_metrics
from jobx.schema import JobPosting

DEFAULT_MODEL = "claude-opus-5-5"
# USD per million tokens: (input, output, cache read). First-party API list prices.
PRICING: dict[str, tuple[float, float, float]] = {
    "claude-fable-5-1": (10.0, 50.0, 0.25),
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}
CACHE_WRITE_MULTIPLIER = 1.25
# Models that take the server-side refusal fallback (`fallbacks: "default"`).
FALLBACK_MODELS = frozenset({"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"})
FALLBACK_BETA = "server-side-fallback-2026-07-01"
SHOT_NAMES = {0: "zero_shot"}


def approach_id(model: str, shots: int) -> str:
    return f"{model}_{SHOT_NAMES.get(shots, 'few_shot')}"


def output_schema() -> dict[str, Any]:
    """JobPosting schema in the form the structured-outputs API accepts (SDK transform)."""
    from anthropic import transform_schema

    schema: dict[str, Any] = transform_schema(JobPosting)
    return schema


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def add(self, other: Usage) -> None:
        for k in asdict(self):
            setattr(self, k, getattr(self, k) + getattr(other, k))

    @classmethod
    def from_response(cls, usage: Any) -> Usage:
        def g(name: str) -> int:
            return int(getattr(usage, name, 0) or 0)

        return cls(
            g("input_tokens"),
            g("output_tokens"),
            g("cache_read_input_tokens"),
            g("cache_creation_input_tokens"),
        )

    def cost(self, model: str) -> float:
        p_in, p_out, p_cache = PRICING.get(model, PRICING[DEFAULT_MODEL])
        return (
            self.input_tokens * p_in
            + self.output_tokens * p_out
            + self.cache_read_input_tokens * p_cache
            + self.cache_creation_input_tokens * p_in * CACHE_WRITE_MULTIPLIER
        ) / 1e6


def build_request(
    model: str,
    text: str,
    shots: Sequence[dict[str, Any]],
    *,
    effort: str,
    max_tokens: int,
    use_fallbacks: bool = True,
) -> dict[str, Any]:
    msgs = prompts.messages(text, shots)
    # The stable prefix (system prompt, then the few-shot turns) gets a cache breakpoint; if the
    # prefix is below the model's minimum cacheable length this is silently a no-op.
    system = [{"type": "text", "text": msgs[0]["content"], "cache_control": {"type": "ephemeral"}}]
    turns: list[dict[str, Any]] = [dict(m) for m in msgs[1:]]
    if shots:
        last_shot = turns[-2]
        last_shot["content"] = [
            {"type": "text", "text": last_shot["content"], "cache_control": {"type": "ephemeral"}}
        ]
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": turns,
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": output_schema()}},
    }
    if use_fallbacks and model in FALLBACK_MODELS:
        kwargs["betas"] = [FALLBACK_BETA]
        kwargs["fallbacks"] = "default"
    return kwargs


def _retry_after(err: Any) -> float | None:
    resp = getattr(err, "response", None)
    headers = getattr(resp, "headers", None)
    if headers is None:
        return None
    try:
        v = headers.get("retry-after")
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


async def call_with_retries(
    create: Callable[..., Awaitable[Any]],
    kwargs: dict[str, Any],
    *,
    max_attempts: int = 6,
    base_delay: float = 2.0,
    max_delay: float = 90.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Any:
    """Retry 429 (honouring retry-after), 5xx/overloaded, timeouts and connection errors; raise others."""
    import anthropic

    for attempt in range(max_attempts):
        try:
            return await create(**kwargs)
        except anthropic.RateLimitError as e:
            if attempt == max_attempts - 1:
                raise
            delay = _retry_after(e) or min(base_delay * 2**attempt, max_delay)
        except (anthropic.APIConnectionError, anthropic.APITimeoutError):
            if attempt == max_attempts - 1:
                raise
            delay = min(base_delay * 2**attempt, max_delay)
        except anthropic.APIStatusError as e:
            if e.status_code < 500 or attempt == max_attempts - 1:
                raise
            delay = _retry_after(e) or min(base_delay * 2**attempt, max_delay)
        await sleep(delay + random.uniform(0, 1))
    raise RuntimeError("unreachable")


@dataclass
class RunSummary:
    approach: str
    split: str
    n_requested: int
    n_done: int = 0
    n_errors: int = 0
    n_refusals: int = 0
    n_fallback_served: int = 0
    skipped_budget: int = 0
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    wall_s: float = 0.0


def _text_of(resp: Any) -> str:
    return "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")


class Budget:
    """Shared across runs: stops starting new requests once spent + in-flight reserve would exceed the cap."""

    def __init__(self, cap_usd: float, reserve_per_call: float) -> None:
        self.cap = cap_usd
        self.reserve = reserve_per_call
        self.spent = 0.0
        self.in_flight = 0

    def try_acquire(self) -> bool:
        if self.spent + (self.in_flight + 1) * self.reserve > self.cap:
            return False
        self.in_flight += 1
        return True

    def release(self, cost: float) -> None:
        self.in_flight -= 1
        self.spent += cost


async def run_split(
    client: Any,
    examples: Sequence[dict[str, Any]],
    *,
    model: str,
    shots: Sequence[dict[str, Any]],
    split: str,
    out_dir: Path,
    budget: Budget,
    effort: str = "low",
    max_tokens: int = 16000,
    concurrency: int = 8,
    use_fallbacks: bool = True,
    meta_extra: dict[str, Any] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[Path, RunSummary]:
    approach = approach_id(model, len(shots))
    summary = RunSummary(approach, split, len(examples))
    sem = asyncio.Semaphore(concurrency)
    create = (
        client.beta.messages.create if use_fallbacks and model in FALLBACK_MODELS else client.messages.create
    )
    rows: list[dict[str, Any] | None] = [None] * len(examples)
    t_start = time.perf_counter()

    async def one(i: int, ex: dict[str, Any]) -> None:
        async with sem:
            if not budget.try_acquire():
                summary.skipped_budget += 1
                return
            cost = 0.0
            t0 = time.perf_counter()
            row: dict[str, Any] = {"id": ex["id"]}
            try:
                kwargs = build_request(
                    model,
                    ex["text"],
                    shots,
                    effort=effort,
                    max_tokens=max_tokens,
                    use_fallbacks=use_fallbacks,
                )
                resp = await call_with_retries(create, kwargs, sleep=sleep)
                usage = Usage.from_response(resp.usage)
                served = str(getattr(resp, "model", model) or model)
                cost = usage.cost(served if served in PRICING else model)
                summary.usage.add(usage)
                summary.cost_usd += cost
                stop = getattr(resp, "stop_reason", None)
                summary.n_refusals += stop == "refusal"
                summary.n_fallback_served += served != model
                row.update(
                    {
                        "output": "" if stop == "refusal" else _text_of(resp),
                        "stop_reason": stop,
                        "served_model": served,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "cost_usd": round(cost, 6),
                    }
                )
            except Exception as e:  # recorded as an empty (invalid) prediction; only the type is kept
                summary.n_errors += 1
                row.update({"output": "", "error": type(e).__name__})
            finally:
                budget.release(cost)
            row["latency_s"] = round(time.perf_counter() - t0, 3)
            rows[i] = row
            summary.n_done += 1

    await asyncio.gather(*(one(i, ex) for i, ex in enumerate(examples)))
    summary.wall_s = round(time.perf_counter() - t_start, 1)
    summary.cost_usd = round(summary.cost_usd, 4)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = out_dir / f"{approach}__{split}.jsonl"
    with open(pred, "w", encoding="utf-8") as f:
        for r in rows:
            if r is not None:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "approach": approach,
        "model": model,
        "decoding": "structured outputs (json_schema)",
        "hardware": "Anthropic API",
        "split": split,
        "n_shots": len(shots),
        "shot_ids": [s["id"] for s in shots],
        "effort": effort,
        "max_tokens": max_tokens,
        "refusal_fallbacks": "default" if use_fallbacks and model in FALLBACK_MODELS else None,
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "usage": asdict(summary.usage),
        "cost_usd": summary.cost_usd,
        "n_errors": summary.n_errors,
        "n_refusals": summary.n_refusals,
        "n_fallback_served": summary.n_fallback_served,
        "skipped_budget": summary.skipped_budget,
        "concurrency": concurrency,
        "wall_s": summary.wall_s,
        **(meta_extra or {}),
    }
    pred.with_name(pred.name.removesuffix(".jsonl") + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    return pred, summary


async def estimate_cost(
    client: Any,
    plan: Sequence[tuple[str, Sequence[dict[str, Any]], Sequence[dict[str, Any]]]],
    *,
    model: str,
    est_output_tokens: int,
    sample: int = 5,
) -> dict[str, Any]:
    """plan: (split, examples, shots). Input tokens measured with count_tokens on a sample per run."""
    p_in, p_out, _ = PRICING.get(model, PRICING[DEFAULT_MODEL])
    runs = []
    total = 0.0
    for split, examples, shots in plan:
        if not examples:
            continue
        probe = list(examples[:: max(1, len(examples) // sample)][:sample])
        counts = []
        for ex in probe:
            msgs = prompts.messages(ex["text"], shots)
            r = await client.messages.count_tokens(
                model=model,
                system=msgs[0]["content"],
                messages=msgs[1:],
                output_config={"format": {"type": "json_schema", "schema": output_schema()}},
            )
            counts.append(int(r.input_tokens))
        mean_in = sum(counts) / len(counts)
        cost = len(examples) * (mean_in * p_in + est_output_tokens * p_out) / 1e6
        total += cost
        runs.append(
            {
                "approach": approach_id(model, len(shots)),
                "split": split,
                "n": len(examples),
                "mean_input_tokens": round(mean_in),
                "est_output_tokens": est_output_tokens,
                "est_cost_usd": round(cost, 4),
            }
        )
    return {
        "model": model,
        "price_per_mtok": {"input": p_in, "output": p_out},
        "runs": runs,
        "total_est_cost_usd": round(total, 4),
    }


async def main_async(a: argparse.Namespace, client: Any = None) -> int:
    root = Path(a.root)
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: ANTHROPIC_API_KEY is not set (add it as a repository secret).", file=sys.stderr)
            return 2
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(max_retries=0)  # retries are handled by call_with_retries
    shots_list = [int(x) for x in str(a.shots).split(",") if x.strip()]
    splits = [s for s in a.splits.split(",") if s.strip()]
    splits = [s for s in splits if (root / SPLIT_FILES[s]).exists()]
    train = read_jsonl(root / SPLIT_FILES["train"])
    pools = {s: read_jsonl(root / SPLIT_FILES[s]) for s in splits}

    def make_plan(n: int) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
        return [
            (split, eval_subset(pools[split], n), prompts.select_few_shot(train, k=k) if k else [])
            for split in splits
            for k in shots_list
        ]

    n = a.max_examples
    plan = make_plan(n)
    est = await estimate_cost(client, plan, model=a.model, est_output_tokens=a.est_output_tokens)
    if est["total_est_cost_usd"] > a.budget_usd and a.fit_budget:
        # Same per-example cost, fewer examples: shrink the evenly spaced subset to fit 90% of the budget.
        per_example = est["total_est_cost_usd"] / max(1, sum(r["n"] for r in est["runs"])) * len(plan)
        n = max(1, int(0.9 * a.budget_usd / per_example))
        plan = make_plan(n)
        est = await estimate_cost(client, plan, model=a.model, est_output_tokens=a.est_output_tokens)
        est["fitted_max_examples"] = n
    print(json.dumps(est, indent=2))
    if a.estimate_only:
        return 0
    if est["total_est_cost_usd"] > a.budget_usd:
        print(
            f"ERROR: estimated ${est['total_est_cost_usd']:.2f} exceeds the ${a.budget_usd:.2f} budget; "
            "lower --max-examples or --shots.",
            file=sys.stderr,
        )
        return 3
    p_in, p_out, _ = PRICING.get(a.model, PRICING[DEFAULT_MODEL])
    reserve = (
        max((r["mean_input_tokens"] for r in est["runs"]), default=0) * p_in + 3 * a.est_output_tokens * p_out
    ) / 1e6
    budget = Budget(a.budget_usd, reserve)
    meta_extra = {"commit": git_commit(root), "estimate_usd": est["total_est_cost_usd"]}
    for split, examples, shots in plan:
        pred, s = await run_split(
            client,
            examples,
            model=a.model,
            shots=shots,
            split=split,
            out_dir=root / "results" / "predictions",
            budget=budget,
            effort=a.effort,
            max_tokens=a.max_tokens,
            concurrency=a.concurrency,
            use_fallbacks=not a.no_fallbacks,
            meta_extra=meta_extra,
        )
        m = write_metrics(pred, root) if s.n_done else {}
        print(
            f"{s.approach}/{split}: done {s.n_done}/{s.n_requested}, errors {s.n_errors}, refusals "
            f"{s.n_refusals}, budget-skipped {s.skipped_budget}, cost ${s.cost_usd:.4f}, "
            f"micro F1 {m.get('micro_f1')}"
        )
    print(f"TOTAL spent ${budget.spent:.4f} (budget ${a.budget_usd:.2f})")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--splits", default="test_synth,test_real", help="comma list; missing splits are skipped")
    ap.add_argument("--shots", default="0,3", help="comma list of shot counts, e.g. 0,3")
    ap.add_argument("--max-examples", type=int, default=100, help="evenly spaced subset per split")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--budget-usd", type=float, default=2.0)
    ap.add_argument(
        "--est-output-tokens",
        type=int,
        default=400,
        help="assumed output tokens per call (JSON + thinking) for the estimate",
    )
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument(
        "--no-fit-budget",
        dest="fit_budget",
        action="store_false",
        help="abort instead of shrinking the subset when the estimate exceeds the budget",
    )
    ap.add_argument("--no-fallbacks", action="store_true", help="disable server-side refusal fallbacks")
    ap.add_argument("--root", default=str(REPO_ROOT))
    return ap


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(main_async(parser().parse_args(argv))))


if __name__ == "__main__":
    main()
