"""Hosted API baseline over any OpenAI-compatible endpoint (default: the free Gemini API tier).

* same system prompt, user template, few-shot exemplars and JSON schema as every other approach;
* structured output when the provider supports it: ``json_schema`` -> ``json_object`` -> plain prompt,
  downgrading automatically on a 400 that rejects ``response_format``; the output is parsed by the normal
  evaluator either way;
* free-tier friendly: client-side rate limiting (default 10 requests/minute), backoff on 429/5xx that
  honours ``retry-after``, an on-disk response cache keyed by the exact request, and resumable runs
  (finished ids in the predictions file are skipped). Repeated 429s are treated as quota exhaustion: the
  run stops cleanly and the next invocation resumes;
* token counts per request; cost from configurable prices (0 on the free tier);
* output: the standard ``results/predictions/<approach>__<split>.jsonl`` + ``.meta.json``.

Providers are presets of (base_url, key env var, default model); ``--base-url/--key-env/--model`` override:

    python -m jobx.baselines.openai_compat_runner --provider gemini --shots 3
    python -m jobx.baselines.openai_compat_runner --provider groq --model llama-3.3-70b-versatile

The key is read from the environment variable and never printed or written.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
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


# Default model ids are the providers' free-tier models at the time of writing; override with --model.
PROVIDERS: dict[str, Provider] = {
    "gemini": Provider(
        "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-2.5-flash"
    ),
    "groq": Provider("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "openrouter": Provider(
        "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free"
    ),
}
RESPONSE_FORMATS: tuple[str, ...] = ("json_schema", "json_object", "none")
SHOT_NAMES = {0: "zero_shot"}


class QuotaExhausted(RuntimeError):
    """429s persisted through every retry: stop and resume later (free-tier daily quota)."""


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
            return d
        except json.JSONDecodeError:
            return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        if self.dir:
            (self.dir / f"{key}.json").write_text(json.dumps(value, ensure_ascii=False))


def _retry_after(err: Any) -> float | None:
    headers = getattr(getattr(err, "response", None), "headers", None)
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
    limiter: RateLimiter,
    max_attempts: int = 6,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Any:
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
        except openai.APIStatusError as e:
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
    response_format: str = ""
    format_downgrades: list[str] = field(default_factory=list)


async def run_split(
    client: Any,
    examples: Sequence[dict[str, Any]],
    *,
    base_url: str,
    model: str,
    approach_prefix: str,
    shots: Sequence[dict[str, Any]],
    split: str,
    out_dir: Path,
    limiter: RateLimiter,
    cache: ResponseCache,
    mode: str = "json_schema",
    max_tokens: int = 1024,
    concurrency: int = 2,
    price_in: float = 0.0,
    price_out: float = 0.0,
    meta_extra: dict[str, Any] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[Path, RunStats]:
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
            while True:
                kwargs = build_request(model, ex["text"], shots, mode=state["mode"], max_tokens=max_tokens)
                ckey = ResponseCache.key(base_url, kwargs)
                hit = cache.get(ckey)
                if hit is not None:
                    stats.n_cached += 1
                    row.update(hit)
                    break
                try:
                    resp = await call_with_retries(
                        client.chat.completions.create, kwargs, limiter=limiter, sleep=sleep
                    )
                except QuotaExhausted:
                    stats.quota_exhausted = True
                    stop.set()
                    return
                except Exception as e:
                    if _is_format_rejection(e) and state["mode"] != "none":
                        nxt = RESPONSE_FORMATS[RESPONSE_FORMATS.index(state["mode"]) + 1]
                        stats.format_downgrades.append(f"{state['mode']}->{nxt}")
                        state["mode"] = nxt
                        continue
                    stats.n_errors += 1
                    row.update({"output": "", "error": type(e).__name__})
                    break
                usage = getattr(resp, "usage", None)
                choice = resp.choices[0]
                value = {
                    "output": choice.message.content or "",
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "served_model": str(getattr(resp, "model", model) or model),
                    "response_format": state["mode"],
                }
                cache.put(ckey, value)
                stats.n_new += 1
                row.update(value)
                break
            row["latency_s"] = round(time.perf_counter() - t0, 3)
            await append(row)

    await asyncio.gather(*(one(e) for e in examples))
    rows = [results[e["id"]] for e in examples if e["id"] in results]
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
        **(meta_extra or {}),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return pred, stats


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
    limiter = RateLimiter(a.rpm)
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
                "rpm": a.rpm,
                "est_minutes_at_rpm": round(n_planned / a.rpm, 1) if a.rpm else None,
                "price_per_mtok": {"input": price_in, "output": price_out},
            },
            indent=2,
        )
    )
    meta_extra = {"commit": git_commit(root), "provider": a.provider}
    exit_code = 0
    for split in splits:
        examples = eval_subset(read_jsonl(root / SPLIT_FILES[split]), a.max_examples)
        for k in shots_list:
            shots = prompts.select_few_shot(train, k=k) if k else []
            pred, st = await run_split(
                client,
                examples,
                base_url=base_url,
                model=model,
                approach_prefix=a.approach_prefix,
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
            m = write_metrics(pred, root) if pred.stat().st_size else {}
            print(
                f"{st.approach}/{split}: new {st.n_new}, cached {st.n_cached}, resumed {st.n_resumed}, "
                f"errors {st.n_errors}, tokens {st.prompt_tokens}+{st.completion_tokens}, cost ${st.cost_usd}, "
                f"format {st.response_format}, micro F1 {m.get('micro_f1')}"
            )
            if st.quota_exhausted:
                print("Quota exhausted (persistent 429): progress saved; rerun later to resume.")
                return 4
    return exit_code


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="gemini", help=f"preset: {', '.join(PROVIDERS)}")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--key-env", default=None, help="name of the env var holding the API key")
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--approach-prefix",
        default="gemini-flash",
        help="results row id prefix, e.g. gemini-flash -> gemini-flash_few_shot",
    )
    ap.add_argument("--splits", default="test_synth,test_real")
    ap.add_argument("--shots", default="3")
    ap.add_argument("--max-examples", type=int, default=200)
    ap.add_argument("--response-format", default="json_schema", choices=RESPONSE_FORMATS)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--rpm", type=float, default=10.0, help="client-side requests per minute")
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
