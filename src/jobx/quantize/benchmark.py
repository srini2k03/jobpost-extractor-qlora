"""CPU benchmark of a GGUF model through llama.cpp's ``llama-server`` (used in GitHub Actions).

Measures, on the same test examples used everywhere else:
* generation speed (tokens/s, from llama.cpp's own timings), mean latency per posting;
* peak resident memory of the server process (VmHWM from /proc);
* accuracy: predictions are written in the standard format and scored with jobx.eval.

    python -m jobx.quantize.benchmark --server-bin llama.cpp/build/bin/llama-server \
        --model gguf/model-Q4_K_M.gguf --variant "GGUF Q4_K_M (llama.cpp)" \
        --approach qwen2.5-1.5b_gguf_q4_k_m --split test_synth --max-examples 150
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jobx import prompts
from jobx.data.build import read_jsonl
from jobx.eval.metrics import aggregate
from jobx.eval.run import SPLIT_FILES, eval_subset, git_commit, score_file


def peak_rss_gb(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024 / 1024, 3)
    except OSError:
        return None
    return None


def wait_ready(url: str, proc: subprocess.Popen[bytes], timeout: float = 600) -> None:
    import httpx

    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with {proc.returncode}")
        try:
            if httpx.get(f"{url}/health", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise TimeoutError("llama-server did not become ready")


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def run(a: argparse.Namespace) -> dict[str, Any]:
    import httpx

    examples = eval_subset(read_jsonl(Path(a.root) / SPLIT_FILES[a.split]), a.max_examples or None)
    shots = (
        prompts.select_few_shot(read_jsonl(Path(a.root) / SPLIT_FILES["train"]), k=a.shots) if a.shots else []
    )
    response_format = None
    if a.guided:
        from jobx.baselines.guided import guided_schema

        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "JobPosting", "schema": guided_schema()},
        }
    port = a.port
    cmd = [
        a.server_bin,
        "-m",
        a.model,
        "-c",
        str(a.ctx),
        "-t",
        str(a.threads),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "-np",
        "1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    pred_path = Path(a.out_pred) / f"{a.approach}__{a.split}.jsonl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    tps: list[float] = []
    lat: list[float] = []
    try:
        wait_ready(url, proc)
        with httpx.Client(timeout=600) as client, open(pred_path, "w", encoding="utf-8") as f:
            for e in examples:
                body: dict[str, Any] = {
                    "messages": prompts.messages(e["text"], shots, system=a.system),
                    "temperature": 0,
                    "max_tokens": a.max_tokens,
                    "cache_prompt": True,  # system prompt (+ shots) prefix is reused across requests
                }
                if response_format is not None:
                    body["response_format"] = response_format
                t0 = time.perf_counter()
                r = client.post(f"{url}/v1/chat/completions", json=body)
                r.raise_for_status()
                dt = time.perf_counter() - t0
                d = r.json()
                text = d["choices"][0]["message"]["content"] or ""
                timings = d.get("timings") or {}
                n_tok = (d.get("usage") or {}).get("completion_tokens") or timings.get("predicted_n") or 0
                speed = timings.get("predicted_per_second") or (n_tok / dt if dt else None)
                if speed:
                    tps.append(float(speed))
                lat.append(dt)
                f.write(
                    json.dumps(
                        {"id": e["id"], "output": text, "latency_s": round(dt, 4), "new_tokens": n_tok},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        peak = peak_rss_gb(proc.pid)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    meta = {
        "approach": a.approach,
        "model": a.model_label or os.path.basename(a.model),
        "gguf": os.path.basename(a.model),
        "decoding": "guided (llama.cpp JSON-schema grammar)" if a.guided else "free",
        "system_prompt": a.system,
        "n_shots": len(shots),
        "shot_ids": [s["id"] for s in shots],
        "hardware": f"CPU ({cpu_name()}, {a.threads} threads, GitHub Actions)",
        "date": date,
        "commit": git_commit(a.root),
        "n_examples": len(examples),
    }
    Path(str(pred_path).removesuffix(".jsonl") + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if not a.variant:
        return {"predictions": str(pred_path), "n": len(examples)}
    scores, _ = score_file(pred_path, a.root)
    agg = aggregate(scores, n_boot=0)
    row = {
        "variant": a.variant,
        "hardware": meta["hardware"],
        "size_gb": round(Path(a.model).stat().st_size / 1e9, 3),
        "micro_f1": agg["micro_f1"],
        "json_validity": agg["json_validity"],
        "tokens_per_s": round(sum(tps) / len(tps), 2) if tps else None,
        "mean_latency_s": round(sum(lat) / len(lat), 3) if lat else None,
        "peak_ram_gb": peak,
        "n_examples": len(examples),
        "split": a.split,
        "date": date,
        "commit": meta["commit"],
    }
    bench = Path(a.out_bench)
    bench.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(bench.read_text()) if bench.exists() else {"rows": []}
    existing["rows"] = [r for r in existing["rows"] if r["variant"] != a.variant] + [row]
    bench.write_text(json.dumps(existing, indent=2) + "\n")
    return row


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default=None, help="benchmark row name; omit to write predictions only")
    ap.add_argument("--model-label", default=None, help="model name recorded in the meta (e.g. the HF id)")
    ap.add_argument("--system", default="full", choices=["full", "compact"])
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument(
        "--guided", action="store_true", help="JSON-schema constrained decoding (llama.cpp grammar)"
    )
    ap.add_argument("--approach", required=True)
    ap.add_argument("--split", default="test_synth")
    ap.add_argument("--max-examples", type=int, default=150, help="0 = the whole split")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out-pred", default="results/predictions")
    ap.add_argument("--out-bench", default="results/benchmarks/gguf_cpu.json")
    a = ap.parse_args(argv)
    print(json.dumps(run(a), indent=2))


if __name__ == "__main__":
    main()
