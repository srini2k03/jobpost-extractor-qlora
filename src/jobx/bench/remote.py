"""What runs inside one benchmark container (Modal in practice, any GPU box in principle).

``run_config`` downloads the weights (into the cached Hugging Face volume), starts the server as a local
subprocess (``vllm serve`` or ``llama-server``), waits for ``/health``, runs the phase-1 load generator
in-process (GPU sampler + ``/metrics`` scraper per level), optionally runs the repo's quality eval against
the same server, stops it, and writes raw CSVs, predictions and a manifest. It never imports Modal, so the
whole flow is tested on CPU with the fake server standing in for the inference server.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import subprocess
import tarfile
import time
import traceback
import urllib.request
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from jobx import REPO_ROOT
from jobx.bench.gpu import GpuSampler
from jobx.bench.loadgen import RunResult, StreamingClient, sweep, write_csv
from jobx.bench.plan import ServeConfig, build_manifest, llama_server_args, vllm_args, write_manifest
from jobx.bench.server_metrics import MetricsScraper
from jobx.bench.workload import build_workload

PORT = 8000


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def gpu_name() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    names = [x.strip() for x in out.stdout.splitlines() if x.strip()]
    return ", ".join(names) or None


def engine_version(cfg: ServeConfig, binary: str = "/app/llama-server") -> str | None:
    if cfg.engine == "vllm":
        try:
            from importlib.metadata import version

            return version("vllm")
        except Exception:
            return None
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (out.stdout + out.stderr).strip().splitlines()
    return next((t for t in text if "version" in t.lower()), text[0] if text else None)


def snapshot_revision(path: str | Path) -> str | None:
    """The commit hash of a Hugging Face cache path (``.../snapshots/<sha>/...``)."""
    parts = Path(path).parts
    if "snapshots" in parts:
        i = parts.index("snapshots")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def prepare_model(cfg: ServeConfig, cache_dir: str | Path) -> dict[str, Any]:
    """Download weights / GGUF / LoRA adapter into the cache volume; returns local paths and revisions."""
    from huggingface_hub import hf_hub_download, snapshot_download

    info: dict[str, Any] = {}
    if cfg.engine == "llama.cpp":
        assert cfg.gguf_file
        path = hf_hub_download(cfg.model, cfg.gguf_file, revision=cfg.revision)
        info.update(gguf_path=path, model_revision=snapshot_revision(path))
    else:
        path = snapshot_download(cfg.model, revision=cfg.revision)
        info.update(model_path=path, model_revision=snapshot_revision(path))
    if cfg.lora:
        dest = Path(cache_dir) / "adapters" / cfg.lora.name
        tarball = dest.with_suffix(".tar.gz")
        if not tarball.exists():
            tarball.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(cfg.lora.url, tarball)  # fixed https URL from the matrix
        if not dest.exists():
            with tarfile.open(tarball) as tf:
                tf.extractall(dest, filter="data")
        info["lora_path"] = str(dest / cfg.lora.subdir) if cfg.lora.subdir else str(dest)
        info["lora_sha256"] = hashlib.sha256(tarball.read_bytes()).hexdigest()
    return info


def server_argv(cfg: ServeConfig, prep: dict[str, Any], port: int = PORT) -> list[str]:
    if cfg.engine == "llama.cpp":
        return llama_server_args(cfg, prep["gguf_path"], port=port)
    revision = prep.get("model_revision")
    if revision and not cfg.revision:
        cfg = ServeConfig.from_dict({**cfg.to_dict(), "revision": revision})  # pin what was downloaded
    return vllm_args(cfg, port=port, lora_path=prep.get("lora_path"))


def wait_healthy(root_url: str, proc: subprocess.Popen[bytes], timeout_s: float = 1200.0) -> float:
    """Poll ``/health`` until it answers 200; returns seconds waited. Raises if the process exits first."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode} before becoming healthy")
        try:
            if httpx.get(f"{root_url}/health", timeout=5, trust_env=False).status_code == 200:
                return time.perf_counter() - t0
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"server not healthy after {timeout_s:.0f} s")


def tail(path: Path, n: int = 80) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def run_load(cfg: ServeConfig, base_url: str, out_dir: Path, *, gpu: bool = True) -> list[Path]:
    """One gzip CSV per workload: the closed-loop and Poisson levels of ``cfg`` at ``cfg.duration_s`` each."""
    written = []
    root_url = base_url.removesuffix("/v1")
    per_level = cfg.max_requests_per_level
    for w in cfg.workloads:
        reqs = build_workload(w, cfg.n_levels * (cfg.warmup + per_level), seed=cfg.seed)

        async def go(workload: str = w, requests: Sequence[Any] = reqs) -> list[RunResult]:
            async with StreamingClient(base_url, cfg.request_model, timeout_s=900) as client:
                return await sweep(
                    client,
                    requests,
                    label=cfg.name,
                    workload=workload,
                    concurrencies=cfg.concurrency,
                    rates=cfg.rates,
                    per_level=per_level,
                    warmup=cfg.warmup,
                    duration_s=cfg.duration_s,
                    seed=cfg.seed,
                    gpu_sampler_factory=(lambda: GpuSampler(interval_s=0.5)) if gpu else None,
                    server_monitor_factory=lambda: MetricsScraper(f"{root_url}/metrics", interval_s=1.0),
                )

        runs = asyncio.run(go())
        written.append(write_csv(out_dir / f"{w}.csv.gz", runs))
    return written


def run_quality(
    cfg: ServeConfig,
    base_url: str,
    *,
    root: Path,
    hardware: str,
    commit: str | None,
    max_examples: int = 0,
    concurrency: int = 32,
) -> list[Path]:
    """The repo's OpenAI-compatible eval runner against the local server: greedy, max_tokens 384, test_synth,
    SYSTEM_PROMPT + 3 shots (compact prompt, zero-shot, for the fine-tune). Writes the standard prediction files
    ``results/predictions/<approach>__test_synth.jsonl`` + ``.meta.json`` under ``root``."""
    from openai import AsyncOpenAI

    from jobx.baselines import openai_compat_runner as runner

    written: list[Path] = []
    for prefix, fmt in cfg.quality_passes():
        argv = [
            "--provider", cfg.engine, "--base-url", base_url, "--key-env", "VLLM_API_KEY",
            "--model", cfg.request_model, "--model-fallbacks=", "--approach-prefix", prefix,
            "--splits", "test_synth", "--shots", str(cfg.quality_shots), "--max-examples", str(max_examples),
            "--response-format", fmt, "--max-tokens", "384", "--rpm", "0", "--concurrency", str(concurrency),
            "--cache-dir", "", "--root", str(root), "--system", cfg.system, "--hardware", hardware,
        ]  # fmt: skip
        if commit:
            argv += ["--commit", commit]
        args = runner.parser().parse_args(argv)
        client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", max_retries=2, timeout=600)
        code = asyncio.run(runner.main_async(args, client=client))
        if code != runner.EXIT_OK:
            raise RuntimeError(f"quality pass {prefix} ({fmt}) exited with {code}")
        approach = runner.approach_id(prefix, cfg.quality_shots)
        pred = root / "results" / "predictions" / f"{approach}__test_synth.jsonl"
        written += [pred, pred.with_name(pred.name.removesuffix(".jsonl") + ".meta.json")]
    return written


def run_config(
    cfg_dict: dict[str, Any],
    *,
    out_root: str | Path,
    cache_dir: str | Path = "/cache",
    usd_per_hour: float = 0.0,
    estimate_minutes: float | None = None,
    commit: str | None = None,
    skip_load: bool = False,
    skip_quality: bool = False,
    quality_max_examples: int = 0,
    server_cmd: Sequence[str] | None = None,
    download: bool = True,
    repo_root: str | Path = REPO_ROOT,
    health_timeout_s: float = 1200.0,
    port: int = PORT,
) -> dict[str, Any]:
    """Run one config end to end. Files land under ``out_root`` at their repo-relative paths
    (``results/bench/raw/<config>/<workload>.csv.gz``, ``results/bench/manifests/<config>.json``,
    ``results/predictions/...``); returns ``{"manifest": ..., "files": {relpath: bytes}}``."""
    cfg = ServeConfig.from_dict(cfg_dict)
    out_root = Path(out_root)
    raw_dir = out_root / "results" / "bench" / "raw" / cfg.name
    log_path = out_root / "results" / "bench" / "logs" / f"{cfg.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    t0 = time.perf_counter()
    prep: dict[str, Any] = {}
    argv: list[str] = list(server_cmd or [])
    files: list[Path] = []
    error: str | None = None
    extra: dict[str, Any] = {}
    proc: subprocess.Popen[bytes] | None = None
    gpu = gpu_name()
    version = engine_version(cfg)
    try:
        if download:
            prep = prepare_model(cfg, cache_dir)
        if not argv:
            argv = server_argv(cfg, prep, port)
        with open(log_path, "wb") as log:
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        root_url = f"http://127.0.0.1:{port}"
        extra["startup_s"] = round(wait_healthy(root_url, proc, health_timeout_s), 1)
        base_url = f"{root_url}/v1"
        if not skip_load:
            files += run_load(cfg, base_url, raw_dir, gpu=gpu is not None)
        if cfg.quality and not skip_quality:
            hardware = f"{gpu or 'GPU'} (Modal), {cfg.engine} {version or '?'}"
            preds = run_quality(
                cfg, base_url, root=Path(repo_root), hardware=hardware, commit=commit,
                max_examples=quality_max_examples,
            )  # fmt: skip
            for p in preds:
                dest = out_root / p.relative_to(repo_root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if p.resolve() != dest.resolve():
                    shutil.copyfile(p, dest)
                files.append(dest)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        extra["traceback"] = traceback.format_exc()[-4000:]
        extra["server_log_tail"] = tail(log_path)
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    manifest = build_manifest(
        cfg,
        argv=argv,
        engine_version=version,
        gpu_name=gpu,
        model_revision=prep.get("model_revision"),
        started_utc=started,
        wall_s=time.perf_counter() - t0,
        usd_per_hour=usd_per_hour,
        files=[str(p.relative_to(out_root)) for p in files],
        estimate_minutes=estimate_minutes,
        error=error,
        extra={**extra, **({"lora_sha256": prep["lora_sha256"]} if "lora_sha256" in prep else {})},
    )
    mpath = write_manifest(out_root / "results" / "bench" / "manifests" / f"{cfg.name}.json", manifest)
    payload = {str(p.relative_to(out_root)): p.read_bytes() for p in [*files, mpath]}
    return {"manifest": manifest, "files": payload}
