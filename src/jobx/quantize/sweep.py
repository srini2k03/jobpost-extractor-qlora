"""Quantisation sweep: every GGUF variant of the fine-tune benchmarked one after another on ONE runner.

    python -m jobx.quantize.sweep --server-bin llama.cpp/build/bin/llama-server --gguf-dir gguf \
        --name jobx-qwen2.5-0.5b --quants F16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_M Q2_K

For each variant: file size; cold start (page cache dropped when possible, server launch -> healthy);
first-request latency; warm latency (mean / p50 / p95); generation and prompt-processing tokens/s from
llama.cpp's own timings; peak resident memory (VmHWM); JSON validity, schema validity, micro-F1, field
exact match and hallucination rate on a fixed evenly spaced test_synth subset and on test_real. A
threads sweep (1/2/4) on one variant measures how speed scales with CPU threads. Everything runs with
the same llama.cpp build, prompt (compact) and greedy decoding, so differences come from quantisation.

Writes ``results/quant/sweep.json`` and the raw outputs under ``results/quant/``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jobx import prompts
from jobx.data.build import read_jsonl
from jobx.eval.metrics import aggregate, score_example
from jobx.eval.run import SPLIT_FILES, eval_subset, git_commit
from jobx.quantize.benchmark import cpu_name, peak_rss_gb, wait_ready


def drop_page_cache() -> bool:
    """Best effort (GitHub runners allow passwordless sudo); returns whether it worked."""
    try:
        subprocess.run(["sync"], check=True)
        r = subprocess.run(
            ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], capture_output=True, check=False
        )
        return r.returncode == 0
    except OSError:
        return False


def _pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(q * len(s)))], 3)


def run_variant(
    server_bin: str,
    model: Path,
    examples: dict[str, list[dict[str, Any]]],
    *,
    threads: int,
    ctx: int,
    port: int,
    out_dir: Path,
    tag: str,
    max_tokens: int = 384,
    batch: int | None = None,
) -> dict[str, Any]:
    import httpx

    cache_dropped = drop_page_cache()
    cmd = [server_bin, "-m", str(model), "-c", str(ctx), "-t", str(threads), "--port", str(port)]
    cmd += ["--host", "127.0.0.1", "-np", "1"]
    if batch:
        cmd += ["-b", str(batch)]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    rows: dict[str, list[dict[str, Any]]] = {}
    try:
        wait_ready(url, proc)
        startup_s = time.perf_counter() - t0
        lat: list[float] = []
        gen_tps: list[float] = []
        pp_tps: list[float] = []
        first_latency = None
        with httpx.Client(timeout=900) as client:
            for split, exs in examples.items():
                rows[split] = []
                for e in exs:
                    body = {
                        "messages": prompts.messages(e["text"], system="compact"),
                        "temperature": 0,
                        "max_tokens": max_tokens,
                        "cache_prompt": True,
                    }
                    t1 = time.perf_counter()
                    r = client.post(f"{url}/v1/chat/completions", json=body)
                    r.raise_for_status()
                    dt = time.perf_counter() - t1
                    d = r.json()
                    tm = d.get("timings") or {}
                    if first_latency is None:
                        first_latency = dt
                    else:
                        lat.append(dt)
                    if tm.get("predicted_per_second"):
                        gen_tps.append(float(tm["predicted_per_second"]))
                    if tm.get("prompt_per_second") and (tm.get("prompt_n") or 0) >= 32:
                        pp_tps.append(float(tm["prompt_per_second"]))
                    rows[split].append(
                        {
                            "id": e["id"],
                            "output": d["choices"][0]["message"]["content"] or "",
                            "latency_s": round(dt, 4),
                            "prompt_tokens": tm.get("prompt_n"),
                            "new_tokens": tm.get("predicted_n"),
                        }
                    )
        peak = peak_rss_gb(proc.pid)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rs in rows.items():
        (out_dir / f"{tag}__{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs)
        )
    return {
        "startup_s": round(startup_s, 3),
        "page_cache_dropped": cache_dropped,
        "first_request_latency_s": round(first_latency or 0.0, 3),
        "warm_latency_s": {
            "mean": round(statistics.mean(lat), 3) if lat else None,
            "p50": _pct(lat, 0.5),
            "p95": _pct(lat, 0.95),
        },
        "gen_tokens_per_s": round(statistics.mean(gen_tps), 2) if gen_tps else None,
        "prompt_tokens_per_s": round(statistics.mean(pp_tps), 1) if pp_tps else None,
        "peak_ram_gb": peak,
        "_rows": rows,
    }


def accuracy(rows: list[dict[str, Any]], gold: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scores = [
        score_example(r["id"], gold[r["id"]]["label"], r["output"], gold[r["id"]].get("label_alternatives"))
        for r in rows
    ]
    m = aggregate(scores, n_boot=0)
    return {
        k: (round(m[k], 4) if isinstance(m[k], float) else m[k])
        for k in ("n", "json_validity", "schema_validity", "micro_f1", "field_em_mean", "hallucination_rate")
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-bin", required=True)
    ap.add_argument("--gguf-dir", required=True)
    ap.add_argument("--name", default="jobx-qwen2.5-0.5b")
    ap.add_argument(
        "--quants", nargs="+", default=["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K"]
    )
    ap.add_argument("--synth-n", type=int, default=150, help="evenly spaced test_synth subset")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--thread-sweep", nargs="*", type=int, default=[1, 2, 4])
    ap.add_argument("--thread-sweep-quant", default="Q4_K_M")
    ap.add_argument("--thread-sweep-n", type=int, default=30)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="results/quant/sweep.json")
    a = ap.parse_args(argv)
    root = Path(a.root)
    gold: dict[str, dict[str, Any]] = {}
    examples: dict[str, list[dict[str, Any]]] = {}
    for split, n in (("test_synth", a.synth_n), ("test_real", 0)):
        p = root / SPLIT_FILES[split]
        if p.exists():
            exs = eval_subset(read_jsonl(p), n or None)
            examples[split] = exs
            gold.update({e["id"]: e for e in exs})
    variants = []
    for q in a.quants:
        model = Path(a.gguf_dir) / f"{a.name}-{q}.gguf"
        res = run_variant(
            a.server_bin,
            model,
            examples,
            threads=a.threads,
            ctx=a.ctx,
            port=a.port,
            out_dir=root / "results/quant",
            tag=q.lower(),
        )
        rows = res.pop("_rows")
        res["accuracy"] = {s: accuracy(rs, gold) for s, rs in rows.items()}
        variants.append({"quant": q, "size_gb": round(model.stat().st_size / 1e9, 3), **res})
        print(json.dumps(variants[-1]), flush=True)
    sweep = []
    if a.thread_sweep:
        model = Path(a.gguf_dir) / f"{a.name}-{a.thread_sweep_quant}.gguf"
        sub = {"test_synth": eval_subset(examples.get("test_synth", []), a.thread_sweep_n)}
        for t in a.thread_sweep:
            res = run_variant(
                a.server_bin,
                model,
                sub,
                threads=t,
                ctx=a.ctx,
                port=a.port,
                out_dir=root / "results/quant/threads",
                tag=f"t{t}",
            )
            res.pop("_rows")
            sweep.append({"quant": a.thread_sweep_quant, "threads": t, **res})
            print(json.dumps(sweep[-1]), flush=True)
    out = {
        "hardware": f"CPU ({cpu_name()}, {a.threads} threads, GitHub Actions)",
        "llama_cpp": os.environ.get("LLAMA_CPP_REF"),
        "ctx": a.ctx,
        "prompt": "compact",
        "decoding": "greedy, free (no grammar)",
        "subsets": {s: len(v) for s, v in examples.items()},
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "commit": git_commit(root),
        "variants": variants,
        "thread_sweep": sweep,
    }
    dest = root / a.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
