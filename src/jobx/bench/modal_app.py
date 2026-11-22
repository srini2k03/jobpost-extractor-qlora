"""Modal app for the serving benchmark: one L4 container per matrix config (configs/bench/matrix.yaml).

    modal run -m jobx.bench.modal_app --config q15-bf16          # one config
    modal run -m jobx.bench.modal_app --matrix all               # every config (refused above the cap ...)
    modal run -m jobx.bench.modal_app --matrix all --yes         # ... unless --yes
    modal run -m jobx.bench.modal_app --request .github/bench-request.json

Needs a Modal account (``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET`` or ``modal token new``). Before anything
starts it prints the GPU-minute and dollar estimate and refuses to run above ``MAX_GPU_MINUTES`` (env var,
default from the matrix: 420) unless ``--yes``. Each container has a timeout of 1.5 x its estimate + 15 min,
and a runtime guard stops launching configs once actual + in-flight minutes would pass the cap.

Each container starts ``vllm serve`` (pinned vLLM, ``vllm==<vllm_version>`` on a CUDA base image) or
``llama-server`` (pinned llama.cpp CUDA image) on localhost, runs the load generator and the quality eval
in-process (:func:`jobx.bench.remote.run_config`), and writes everything to the ``jobx-bench-results``
Volume; Hugging Face weights are cached in the ``jobx-hf-cache`` Volume. Results are returned and written
into this checkout under ``results/`` at their usual paths.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

import modal

from jobx import REPO_ROOT
from jobx.bench import plan

MATRIX = plan.load_matrix()
VLLM_VERSION = str(MATRIX["vllm_version"])
LLAMA_CPP_IMAGE = str(MATRIX["llama_cpp_image"])
CUDA_BASE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
GPU = str(MATRIX.get("gpu", "L4"))
CPU = float((MATRIX.get("container") or {}).get("cpu", 4))
MEMORY_MIB = int(float((MATRIX.get("container") or {}).get("memory_gib", 16)) * 1024)
MAX_GPU_MINUTES = float(os.environ.get("MAX_GPU_MINUTES") or MATRIX.get("max_gpu_minutes", 420))
FUNCTION_TIMEOUT_S = 6 * 3600  # upper bound; each call narrows it with .with_options(timeout=...)
HARNESS_PIP = [
    "httpx>=0.27",
    "numpy>=1.26",
    "pyyaml>=6.0",
    "pydantic>=2.6,<3",
    "typer>=0.12",
    "rich>=13",
    "openai>=1.40",
    "huggingface_hub>=0.30",
    "nvidia-ml-py",
]

app = modal.App("jobx-bench")
hf_cache = modal.Volume.from_name("jobx-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("jobx-bench-results", create_if_missing=True)
VOLUMES: dict[str | PurePosixPath, Any] = {"/cache": hf_cache, "/results": results_volume}


def _with_repo(image: modal.Image) -> modal.Image:
    """The harness code, configs and processed data on top of an engine image (added last: no rebuilds)."""
    return (
        image.env({"HF_HOME": "/cache/hf", "PYTHONPATH": "/root", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
        .add_local_dir(REPO_ROOT / "configs", "/root/configs")
        .add_local_dir(REPO_ROOT / "data" / "processed", "/root/data/processed")
        .add_local_python_source("jobx")
    )


vllm_image = _with_repo(
    modal.Image.from_registry(CUDA_BASE, add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}", *HARNESS_PIP)
)
llama_cpp_image = _with_repo(
    modal.Image.from_registry(LLAMA_CPP_IMAGE, add_python="3.12").entrypoint([]).uv_pip_install(*HARNESS_PIP)
)


def _run(cfg_dict: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
    from jobx.bench.remote import run_config

    out = run_config(cfg_dict, out_root=f"/results/{opts.pop('stamp')}", cache_dir="/cache", **opts)
    hf_cache.commit()
    results_volume.commit()
    return out


@app.function(
    image=vllm_image, gpu=GPU, cpu=CPU, memory=MEMORY_MIB, volumes=VOLUMES, timeout=FUNCTION_TIMEOUT_S
)
def bench_vllm(cfg_dict: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
    """One vLLM config: serve, load test, quality eval, manifest."""
    return _run(cfg_dict, opts)


@app.function(
    image=llama_cpp_image, gpu=GPU, cpu=CPU, memory=MEMORY_MIB, volumes=VOLUMES, timeout=FUNCTION_TIMEOUT_S
)
def bench_llama_cpp(cfg_dict: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
    """The llama.cpp baseline: llama-server with a GGUF, same load generator and eval."""
    return _run(cfg_dict, opts)


def write_payload(files: dict[str, bytes], out: Path) -> list[Path]:
    written = []
    for rel, data in files.items():
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        written.append(dest)
    return written


def _git_commit() -> str | None:
    from jobx.eval.run import git_commit

    return git_commit(REPO_ROOT)


@app.local_entrypoint()
def main(
    config: str = "",
    matrix: str = "",
    request: str = "",
    levels: str = "",
    yes: bool = False,
    max_gpu_minutes: float = 0.0,
    max_parallel: int = 4,
    skip_load: bool = False,
    skip_quality: bool = False,
    quality_max_examples: int = 0,
    out: str = ".",
    dry_run: bool = False,
) -> None:
    """Estimate, check the budget, then run the selected configs (``--matrix all`` = every config)."""
    req = json.loads(Path(request).read_text()) if request else None
    sel = plan.plan_selection(
        MATRIX,
        config=config or matrix,
        request=req,
        levels=json.loads(levels) if levels else None,
        yes=yes,
        cap_minutes=max_gpu_minutes or MAX_GPU_MINUTES,
    )
    print(plan.format_estimate(sel.decision, GPU))
    if not sel.decision.ok:
        raise SystemExit(2)
    if dry_run:
        return
    skip_load = skip_load or bool(sel.request and sel.request.skip_load)
    skip_quality = skip_quality or bool(sel.request and sel.request.skip_quality)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rate = plan.hourly_rate(MATRIX)["total"]
    estimates = {e["config"]: float(e["minutes"]) for e in sel.decision.estimates}
    commit = _git_commit()
    out_dir = Path(out)
    pool = ThreadPoolExecutor(max_workers=max(1, max_parallel))

    def call(cfg: plan.ServeConfig) -> dict[str, Any]:
        fn = bench_llama_cpp if cfg.engine == "llama.cpp" else bench_vllm
        opts = {
            "stamp": stamp,
            "usd_per_hour": rate,
            "estimate_minutes": estimates[cfg.name],
            "commit": commit,
            "skip_load": skip_load,
            "skip_quality": skip_quality,
            "quality_max_examples": quality_max_examples,
        }
        print(f"[{cfg.name}] starting (estimate {estimates[cfg.name]} min)", flush=True)
        res = fn.with_options(timeout=plan.timeout_s(estimates[cfg.name])).remote(cfg.to_dict(), opts)
        write_payload(res["files"], out_dir)
        m: dict[str, Any] = res["manifest"]
        print(
            f"[{cfg.name}] {m['status']} in {m['gpu_minutes']} min, ~${m['estimated_cost_usd']}", flush=True
        )
        return m

    def submit(cfg: plan.ServeConfig) -> Future[dict[str, Any]]:
        return pool.submit(call, cfg)

    cap = (
        max(sel.decision.cap_minutes, sel.decision.total_minutes)
        if sel.decision.ok
        else sel.decision.cap_minutes
    )
    manifests, skipped = plan.run_scheduled(
        sel.configs, estimates, cap_minutes=cap, max_parallel=max_parallel, submit=submit
    )
    pool.shutdown()
    used = sum(float(m.get("gpu_minutes", 0)) for m in manifests)
    print(f"done: {len(manifests)} configs, {used:.1f} GPU-min; skipped (budget): {skipped or 'none'}")
    if any(m.get("status") != "ok" for m in manifests):
        for m in manifests:
            if m.get("status") != "ok":
                print(f"[{m.get('config')}] ERROR {m.get('error')}")
        raise SystemExit(1)
