"""Hosted API baseline over any OpenAI-compatible endpoint (default: GPT-OSS-120B on Groq's free tier).

* same system prompt, user template, few-shot exemplars and JSON schema as every other approach;
* structured output when the provider supports it: ``json_schema`` -> ``json_object`` -> plain prompt,
  downgrading automatically on a 400 that rejects ``response_format``; the output is parsed by the normal
  evaluator either way;
* free-tier friendly: client-side rate limiting (preset default: 25 requests/minute on Groq), backoff on 429/5xx that
  honours ``retry-after``, an on-disk response cache keyed by the exact request, and resumable runs
  (finished ids in the predictions file are skipped). Repeated 429s are treated as quota exhaustion: the
  run stops cleanly and the next invocation resumes;
* token counts per request; cost from configurable prices (0 on the free tier);
* output: the standard ``results/predictions/<approach>__<split>.jsonl`` + ``.meta.json``.

Providers are presets of (base_url, key env var, default model); ``--base-url/--key-env/--model`` override:

    python -m jobx.baselines.openai_compat_runner --provider groq --shots 3      # default
    python -m jobx.baselines.openai_compat_runner --check-complete            # exit 0 if nothing is left
    python -m jobx.baselines.openai_compat_runner --provider gemini --model gemini-3.8-flash

The key is read from the environment variable and never printed or written.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jobx import REPO_ROOT, prompts
from jobx.baselines.guided import inline_refs
from jobx.data.build import read_jsonl
from jobx.eval.run import SPLIT_FILES, eval_subset, git_commit, write_metrics
from jobx.schema import json_schema


@dataclass(frozen=True)
class Provider:
    base_url: str
    key_env: str
    default_model: str
    price_in: float = 0.0  # USD per million tokens
    price_out: float = 0.0
    default_fallbacks: tuple[str, ...] = ()
    default_rpm: float = 10.0
    approach_prefix: str = "api"


# Default model ids are the providers' free-tier models at the time of writing; override with --model.
# Groq is the default: GPT-OSS-120B (OpenAI open-weight, ~117B-parameter MoE) on a free tier that is usable
# from CI. Gemini's free tier had tiny daily quotas and frequent 503s there, and llama-3.3-70b-versatile
# returned 404 for the project's Groq key.
PROVIDERS: dict[str, Provider] = {
    "groq": Provider(
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
        "openai/gpt-oss-120b",
        default_fallbacks=("qwen/qwen3.8-27b",),
        default_rpm=25.0,
        approach_prefix="gpt-oss-120b-groq",
    ),
    "gemini": Provider(
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-3.8-flash",
        default_fallbacks=("gemini-3.7-flash", "gemini-3.5-flash"),
        default_rpm=10.0,
        approach_prefix="gemini-flash",
    ),
    "openrouter": Provider(
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "meta-llama/llama-3.3-70b-instruct:free",
        default_rpm=10.0,
        approach_prefix="openrouter",
    ),
}
DEFAULT_PROVIDER = "groq"
RESPONSE_FORMATS: tuple[str, ...] = ("json_schema", "json_object", "none")
SHOT_NAMES = {0: "zero_shot"}


class QuotaExhausted(RuntimeError):
    """429s persisted through every retry: stop and resume later (free-tier daily quota)."""


class FatalConfigError(RuntimeError):
    """401/403/404: wrong key, no access or unknown model. Never retried; the run aborts without writing."""


class ModelUnavailable(FatalConfigError):
    """403/404 for one model: dropped from the fallback chain (fatal only when no model is left)."""


class ModelOverloaded(RuntimeError):
    """503 on one model: counted by the chain, which moves on after consecutive overloads."""


class ModelChain:
    """Primary model plus fallbacks.

    After ``switch_after`` consecutive 503s a model cools down for ``cooldown_s`` and the next one is used;
    when its cooldown ends the earlier (preferred) model is tried again. 403/404 drop a model for good.
    """

    def __init__(
        self,
        models: Sequence[str],
        *,
        switch_after: int = 2,
        cooldown_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not models:
            raise ValueError("need at least one model")
        self.models = list(dict.fromkeys(models))
        self.switch_after = switch_after
        self.cooldown_s = cooldown_s
        self.clock = clock
        self.dropped: dict[str, str] = {}
        self.cool_until: dict[str, float] = dict.fromkeys(self.models, 0.0)
        self.overloads: dict[str, int] = dict.fromkeys(self.models, 0)
        self.switches: list[str] = []

    @property
    def alive(self) -> list[str]:
        return [m for m in self.models if m not in self.dropped]

    def current(self) -> tuple[str, float]:
        """(model to use, seconds to wait first). Raises FatalConfigError when every model is dropped."""
        alive = self.alive
        if not alive:
            raise FatalConfigError(
                "every model in the chain is unavailable: "
                + "; ".join(f"{m}: {r}" for m, r in self.dropped.items())
            )
        now = self.clock()
        for m in alive:
            if self.cool_until[m] <= now:
                return m, 0.0
        m = min(alive, key=lambda x: self.cool_until[x])
        return m, self.cool_until[m] - now

    def record_success(self, model: str) -> None:
        self.overloads[model] = 0

    def record_overload(self, model: str) -> None:
        self.overloads[model] += 1
        if self.overloads[model] >= self.switch_after:
            self.overloads[model] = 0
            self.cool_until[model] = self.clock() + self.cooldown_s
            self.switches.append(f"{model} overloaded x{self.switch_after}: cooldown {self.cooldown_s:.0f}s")

    def drop(self, model: str, reason: str) -> None:
        self.dropped.setdefault(model, reason)


EXIT_OK, EXIT_QUOTA_PARTIAL, EXIT_FATAL, EXIT_NOTHING_COMPLETED = 0, 4, 5, 6


def approach_id(prefix: str, shots: int) -> str:
    return f"{prefix}_{SHOT_NAMES.get(shots, 'few_shot')}"


def provider_schema() -> dict[str, Any]:
    """Ref-free JobPosting schema without numeric bounds (widest provider compatibility)."""

    def strip(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                k: strip(v)
                for k, v in node.items()
                if k not in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "default", "title")
            }
        if isinstance(node, list):
            return [strip(x) for x in node]
        return node

    out: dict[str, Any] = strip(inline_refs(json_schema()))
    out.pop("$schema", None)
    return out


def response_format(mode: str) -> dict[str, Any] | None:
    if mode == "json_schema":
        return {"type": "json_schema", "json_schema": {"name": "JobPosting", "schema": provider_schema()}}
    if mode == "json_object":
        return {"type": "json_object"}
    return None


def build_request(
    model: str, text: str, shots: Sequence[dict[str, Any]], *, mode: str, max_tokens: int
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": prompts.messages(text, shots),
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    rf = response_format(mode)
    if rf is not None:
        kwargs["response_format"] = rf
    return kwargs


class RateLimiter:
    """Spaces request starts at least 60/rpm seconds apart (shared by all tasks)."""

    def __init__(
        self,
        rpm: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.clock = clock
        self.sleep = sleep
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = self.clock()
            if self._next > now:
                await self.sleep(self._next - now)
                now = self._next
            self._next = now + self.interval


class ResponseCache:
    """One JSON file per request hash under ``cache_dir`` (stores text + usage only, never headers)."""

    def __init__(self, cache_dir: str | Path | None) -> None:
        self.dir = Path(cache_dir) if cache_dir else None
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(base_url: str, kwargs: dict[str, Any]) -> str:
        blob = json.dumps({"base_url": base_url, **kwargs}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.dir:
            return None
        p = self.dir / f"{key}.json"
        if not p.exists():
            return None
        try:
            d: dict[str, Any] = json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
        return d if self.is_success(d) else None  # never replay an error, even from an old cache

    @staticmethod
    def is_success(value: dict[str, Any]) -> bool:
        return (
            not value.get("error") and isinstance(value.get("output"), str) and bool(value["output"].strip())
        )

    def put(self, key: str, value: dict[str, Any]) -> None:
        if self.dir and self.is_success(value):
            (self.dir / f"{key}.json").write_text(json.dumps(value, ensure_ascii=False))


def _msg(err: Any) -> str:
    return str(getattr(err, "message", "") or err)[:300]


def parse_duration(v: str) -> float | None:
    """'7', '1.5', '350ms', '2m59.56s', '1h2m' -> seconds (Groq's x-ratelimit-reset-* format)."""
    v = v.strip()
    try:
        return float(v)
    except ValueError:
        pass
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", v)
    if not parts or "".join(n + u for n, u in parts) != v:
        return None
    scale = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return sum(float(n) * scale[u] for n, u in parts)


def _retry_after(err: Any) -> float | None:
    """Seconds to wait from Retry-After, else Groq's x-ratelimit-reset-requests/-tokens headers."""
    headers = getattr(getattr(err, "response", None), "headers", None)
    if headers is None:
        return None
    for name in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        v = headers.get(name)
        if v is not None:
            d = parse_duration(str(v))
            if d is not None:
                return d
    return None


async def call_with_retries(
    create: Callable[..., Awaitable[Any]],
    kwargs: dict[str, Any],
    *,
    limiter: RateLimiter,
    max_attempts: int = 6,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    overload_raises: bool = False,
) -> Any:
    """Retries 429/5xx/connection errors on one model. 401 -> FatalConfigError; 403/404 -> ModelUnavailable;
    with ``overload_raises`` a 503 is raised as ModelOverloaded so a model chain can switch models."""
    import openai

    for attempt in range(max_attempts):
        await limiter.wait()
        try:
            return await create(**kwargs)
        except openai.RateLimitError as e:
            if attempt == max_attempts - 1:
                raise QuotaExhausted("rate limit persisted through all retries") from e
            delay = _retry_after(e) or min(base_delay * 2**attempt, max_delay)
        except (openai.APIConnectionError, openai.APITimeoutError):
            if attempt == max_attempts - 1:
                raise
            delay = min(base_delay * 2**attempt, max_delay)
        except openai.AuthenticationError as e:
            raise FatalConfigError(f"HTTP 401 (AuthenticationError): {_msg(e)}") from e
        except (openai.PermissionDeniedError, openai.NotFoundError) as e:
            raise ModelUnavailable(f"HTTP {e.status_code} ({type(e).__name__}): {_msg(e)}") from e
        except openai.APIStatusError as e:
            if e.status_code == 503 and overload_raises:
                raise ModelOverloaded(_msg(e)) from e
            if e.status_code < 500 or attempt == max_attempts - 1:
                raise
            delay = _retry_after(e) or min(base_delay * 2**attempt, max_delay)
        await sleep(delay + random.uniform(0, 1))
    raise RuntimeError("unreachable")


def _is_format_rejection(err: Exception) -> bool:
    import openai

    if not isinstance(err, openai.BadRequestError):
        return False
    msg = str(getattr(err, "message", "") or err).lower()
    return any(t in msg for t in ("response_format", "json_schema", "schema", "json mode", "response_mime"))


@dataclass
class RunStats:
    approach: str
    split: str
    n_requested: int
    n_new: int = 0
    n_cached: int = 0
    n_resumed: int = 0
    n_errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    quota_exhausted: bool = False
    n_completed: int = 0
    response_format: str = ""
    format_downgrades: list[str] = field(default_factory=list)


def models_used(rows: Sequence[dict[str, Any]], default: str) -> dict[str, int]:
    """Which chain model answered each completed prediction (count per model, most used first)."""
    c: dict[str, int] = {}
    for r in rows:
        if not r.get("error"):
            m = str(r.get("model") or default)
            c[m] = c.get(m, 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


async def run_split(
    client: Any,
    examples: Sequence[dict[str, Any]],
    *,
    base_url: str,
    model: str | ModelChain,
    approach_prefix: str,
    shots: Sequence[dict[str, Any]],
    split: str,
    out_dir: Path,
    limiter: RateLimiter,
    cache: ResponseCache,
    max_overloads_per_example: int = 8,
    mode: str = "json_schema",
    max_tokens: int = 1024,
    concurrency: int = 2,
    price_in: float = 0.0,
    price_out: float = 0.0,
    meta_extra: dict[str, Any] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[Path | None, RunStats]:
    """Returns (predictions path, stats); the path is None when no request completed."""
    chain = model if isinstance(model, ModelChain) else ModelChain([model])
    model = chain.models[0]  # primary: identifies the run for resuming
    approach = approach_id(approach_prefix, len(shots))
    stats = RunStats(approach, split, len(examples))
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = out_dir / f"{approach}__{split}.jsonl"
    meta_path = pred.with_name(pred.name.removesuffix(".jsonl") + ".meta.json")
    done: dict[str, dict[str, Any]] = {}
    if pred.exists():
        for r in read_jsonl(pred):
            if not r.get("error"):
                done[r["id"]] = r
    prev_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if prev_meta.get("model") not in (None, model):
        done = {}  # a different model: start over
    stats.n_resumed = sum(1 for e in examples if e["id"] in done)
    state = {"mode": prev_meta.get("response_format", mode) if done else mode}
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    stop = asyncio.Event()
    fatal: list[str] = []
    results: dict[str, dict[str, Any]] = dict(done)

    async def append(row: dict[str, Any]) -> None:
        async with lock:
            results[row["id"]] = row

    async def one(ex: dict[str, Any]) -> None:
        if ex["id"] in done:
            return
        async with sem:
            if stop.is_set():
                return
            t0 = time.perf_counter()
            row: dict[str, Any] = {"id": ex["id"]}
            overloads = 0
            while True:
                try:
                    use, wait = chain.current()
                except FatalConfigError as e:
                    fatal.append(str(e))
                    stop.set()
                    return
                kwargs = build_request(use, ex["text"], shots, mode=state["mode"], max_tokens=max_tokens)
                ckey = ResponseCache.key(base_url, kwargs)
                hit = cache.get(ckey)
                if hit is not None:
                    stats.n_cached += 1
                    row.update({"model": use, **hit})
                    break
                if wait > 0:
                    await sleep(wait)
                try:
                    resp = await call_with_retries(
                        client.chat.completions.create,
                        kwargs,
                        limiter=limiter,
                        sleep=sleep,
                        overload_raises=True,
                    )
                except QuotaExhausted:
                    stats.quota_exhausted = True
                    stop.set()
                    return
                except ModelOverloaded:
                    chain.record_overload(use)
                    overloads += 1
                    if overloads >= max_overloads_per_example:
                        stats.n_errors += 1
                        row.update({"output": "", "error": "ModelOverloaded", "model": use})
                        break
                    await sleep(2.0 + random.uniform(0, 1))
                    continue
                except ModelUnavailable as e:
                    chain.drop(use, str(e))
                    continue
                except FatalConfigError as e:
                    fatal.append(str(e))
                    stop.set()
                    return
                except Exception as e:
                    if _is_format_rejection(e) and state["mode"] != "none":
                        nxt = RESPONSE_FORMATS[RESPONSE_FORMATS.index(state["mode"]) + 1]
                        stats.format_downgrades.append(f"{state['mode']}->{nxt}")
                        state["mode"] = nxt
                        continue
                    stats.n_errors += 1
                    row.update({"output": "", "error": type(e).__name__, "model": use})
                    break
                chain.record_success(use)
                usage = getattr(resp, "usage", None)
                choice = resp.choices[0]
                value = {
                    "output": choice.message.content or "",
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "served_model": str(getattr(resp, "model", use) or use),
                    "response_format": state["mode"],
                }
                cache.put(ckey, value)
                stats.n_new += 1
                row.update({"model": use, **value})
                break
            row["latency_s"] = round(time.perf_counter() - t0, 3)
            await append(row)

    await asyncio.gather(*(one(e) for e in examples))
    if fatal:
        raise FatalConfigError(fatal[0])
    rows = [results[e["id"]] for e in examples if e["id"] in results]
    stats.n_completed = sum(1 for r in rows if not r.get("error"))
    if stats.n_completed == 0:
        return None, stats  # nothing to score: leave any previous files untouched
    with open(pred, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats.prompt_tokens = sum(int(r.get("prompt_tokens", 0)) for r in rows)
    stats.completion_tokens = sum(int(r.get("completion_tokens", 0)) for r in rows)
    stats.cost_usd = round((stats.prompt_tokens * price_in + stats.completion_tokens * price_out) / 1e6, 4)
    stats.response_format = state["mode"]
    meta = {
        "approach": approach,
        "model": model,
        "base_url": base_url,
        "hardware": "API",
        "decoding": f"greedy (temperature 0), response_format={state['mode']}",
        "response_format": state["mode"],
        "split": split,
        "n_shots": len(shots),
        "shot_ids": [s["id"] for s in shots],
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "n_examples": len(examples),
        "n_completed": sum(1 for r in rows if not r.get("error")),
        "complete": len(rows) == len(examples) and not any(r.get("error") for r in rows),
        "usage": {
            "requests": len(rows),
            "prompt_tokens": stats.prompt_tokens,
            "completion_tokens": stats.completion_tokens,
        },
        "price_per_mtok": {"input": price_in, "output": price_out},
        "cost_usd": stats.cost_usd,
        "format_downgrades": stats.format_downgrades,
        "model_chain": chain.models,
        "models_used": models_used(rows, model),
        "dropped_models": chain.dropped,
        "model_switches": chain.switches[:50],
        **(meta_extra or {}),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return pred, stats


EXIT_INCOMPLETE = 10


def check_complete(
    root: Path, splits_arg: str, shots_arg: str, prefix: str, model: str, max_examples: int | None
) -> int:
    """Exit code 0 when every (split, shots) run already has an error-free prediction for every example of
    the evenly spaced subset, from the same primary model; EXIT_INCOMPLETE otherwise. No API calls."""
    status = []
    for split in [s.strip() for s in splits_arg.split(",") if s.strip()]:
        if not (root / SPLIT_FILES[split]).exists():
            continue
        wanted = {e["id"] for e in eval_subset(read_jsonl(root / SPLIT_FILES[split]), max_examples)}
        for k in [int(x) for x in str(shots_arg).split(",") if x.strip()]:
            pred = root / "results" / "predictions" / f"{approach_id(prefix, k)}__{split}.jsonl"
            meta_path = pred.with_name(pred.name.removesuffix(".jsonl") + ".meta.json")
            done: set[str] = set()
            if (
                pred.exists()
                and meta_path.exists()
                and json.loads(meta_path.read_text()).get("model") == model
            ):
                done = {r["id"] for r in read_jsonl(pred) if not r.get("error")}
            status.append({"run": pred.name, "done": len(wanted & done), "wanted": len(wanted)})
    complete = bool(status) and all(s["done"] == s["wanted"] for s in status)
    print(json.dumps({"complete": complete, "runs": status}, indent=2))
    return EXIT_OK if complete else EXIT_INCOMPLETE


async def main_async(a: argparse.Namespace, client: Any = None) -> int:
    prov = PROVIDERS.get(a.provider)
    base_url = a.base_url or (prov.base_url if prov else "")
    key_env = a.key_env or (prov.key_env if prov else "")
    model = a.model or (prov.default_model if prov else "")
    if not (base_url and key_env and model):
        print("ERROR: need --provider or all of --base-url, --key-env and --model.", file=sys.stderr)
        return 2
    price_in = a.price_in if a.price_in is not None else (prov.price_in if prov else 0.0)
    price_out = a.price_out if a.price_out is not None else (prov.price_out if prov else 0.0)
    if a.check_complete:
        prefix = a.approach_prefix or (prov.approach_prefix if prov else "api")
        return check_complete(Path(a.root), a.splits, a.shots, prefix, model, a.max_examples)
    if client is None:
        key = os.environ.get(key_env, "")
        if not key:
            print(f"ERROR: {key_env} is not set (add it as a repository secret).", file=sys.stderr)
            return 2
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=base_url, api_key=key, max_retries=0, timeout=120)
    root = Path(a.root)
    shots_list = [int(x) for x in str(a.shots).split(",") if x.strip()]
    splits = [s for s in a.splits.split(",") if s.strip() and (root / SPLIT_FILES[s.strip()]).exists()]
    train = read_jsonl(root / SPLIT_FILES["train"])
    rpm = a.rpm if a.rpm is not None else (prov.default_rpm if prov else 10.0)
    prefix = a.approach_prefix or (prov.approach_prefix if prov else "api")
    limiter = RateLimiter(rpm)
    cache = ResponseCache(a.cache_dir)
    n_planned = sum(
        min(len(read_jsonl(root / SPLIT_FILES[s])), a.max_examples or 10**9) for s in splits
    ) * len(shots_list)
    print(
        json.dumps(
            {
                "provider": a.provider,
                "model": model,
                "base_url": base_url,
                "splits": splits,
                "shots": shots_list,
                "max_requests": n_planned,
                "rpm": rpm,
                "est_minutes_at_rpm": round(n_planned / rpm, 1) if rpm else None,
                "price_per_mtok": {"input": price_in, "output": price_out},
            },
            indent=2,
        )
    )
    fallbacks = [
        m.strip()
        for m in (
            a.model_fallbacks
            if a.model_fallbacks is not None
            else ",".join(prov.default_fallbacks if prov else ())
        ).split(",")
        if m.strip()
    ]
    chain = ModelChain([model, *fallbacks], switch_after=a.switch_after, cooldown_s=a.cooldown)
    print(f"model chain: {' -> '.join(chain.models)}")
    meta_extra = {"commit": git_commit(root), "provider": a.provider}
    total_completed = 0
    for split in splits:
        examples = eval_subset(read_jsonl(root / SPLIT_FILES[split]), a.max_examples)
        for k in shots_list:
            shots = prompts.select_few_shot(train, k=k) if k else []
            try:
                pred, st = await run_split(
                    client,
                    examples,
                    base_url=base_url,
                    model=chain,
                    approach_prefix=prefix,
                    shots=shots,
                    split=split,
                    out_dir=root / "results" / "predictions",
                    limiter=limiter,
                    cache=cache,
                    mode=a.response_format,
                    max_tokens=a.max_tokens,
                    concurrency=a.concurrency,
                    price_in=price_in,
                    price_out=price_out,
                    meta_extra=meta_extra,
                )
            except FatalConfigError as e:
                print(
                    f"FATAL: {e}\nCheck the model id (--model / workflow input) and the API key; "
                    "nothing was written.",
                    file=sys.stderr,
                )
                return EXIT_FATAL
            total_completed += st.n_completed
            m = write_metrics(pred, root) if pred is not None else {}
            print(
                f"{st.approach}/{split}: new {st.n_new}, cached {st.n_cached}, resumed {st.n_resumed}, "
                f"errors {st.n_errors}, tokens {st.prompt_tokens}+{st.completion_tokens}, cost ${st.cost_usd}, "
                f"format {st.response_format}, micro F1 {m.get('micro_f1')}"
            )
            if st.quota_exhausted:
                print("Quota exhausted (persistent 429): progress saved; rerun later to resume.")
                return EXIT_QUOTA_PARTIAL if total_completed else EXIT_NOTHING_COMPLETED
    if total_completed == 0:
        print("ERROR: no request completed; nothing to score.", file=sys.stderr)
        return EXIT_NOTHING_COMPLETED
    return EXIT_OK


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default=DEFAULT_PROVIDER, help=f"preset: {', '.join(PROVIDERS)}")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--key-env", default=None, help="name of the env var holding the API key")
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--model-fallbacks",
        default=None,
        help="comma list tried in order when the model is overloaded (503) or unavailable (403/404); "
        "default: the provider preset's list, '' for none",
    )
    ap.add_argument("--switch-after", type=int, default=2, help="consecutive 503s before switching model")
    ap.add_argument("--cooldown", type=float, default=300.0, help="seconds an overloaded model is skipped")
    ap.add_argument(
        "--approach-prefix",
        default=None,
        help="results row id prefix (default: the preset's, e.g. gpt-oss-120b-groq -> gpt-oss-120b-groq_few_shot)",
    )
    ap.add_argument("--splits", default="test_synth,test_real")
    ap.add_argument("--shots", default="3")
    ap.add_argument("--max-examples", type=int, default=200)
    ap.add_argument("--response-format", default="json_schema", choices=RESPONSE_FORMATS)
    ap.add_argument(
        "--check-complete",
        action="store_true",
        help="no API calls: exit 0 if all planned predictions exist, 10 if work is left",
    )
    ap.add_argument("--max-tokens", type=int, default=2048, help="room for reasoning-model fallbacks")
    ap.add_argument(
        "--rpm", type=float, default=None, help="client-side requests per minute (default: preset)"
    )
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--cache-dir", default=".cache/jobx-api")
    ap.add_argument("--price-in", type=float, default=None, help="USD per million input tokens")
    ap.add_argument("--price-out", type=float, default=None, help="USD per million output tokens")
    ap.add_argument("--root", default=str(REPO_ROOT))
    return ap


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(main_async(parser().parse_args(argv))))


if __name__ == "__main__":
    main()
