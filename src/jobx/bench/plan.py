"""The benchmark matrix as data: expansion, server flags, cost/time estimate, budget, request file, manifest.

Pure Python (no Modal, no GPU) so all of it is unit-tested; :mod:`jobx.bench.modal_app` and
:mod:`jobx.bench.remote` only call into it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from jobx import REPO_ROOT
from jobx.bench.workload import WORKLOADS

DEFAULT_MATRIX = REPO_ROOT / "configs" / "bench" / "matrix.yaml"
ENGINES = ("vllm", "llama.cpp")
# Prompt-token budget per prompt style (measured ~1.8K tokens for SYSTEM_PROMPT + 3 shots + posting; the
# compact prompt + posting is ~250) and max output tokens per workload kind; max_model_len = sum, rounded up.
PROMPT_BUDGET = {"full": 2048, "compact": 512}
OUTPUT_BUDGET = {False: 384, True: 1024}  # synthetic_output -> cap of the lognormal


@dataclass
class LoraSpec:
    name: str
    url: str
    subdir: str = ""
    max_lora_rank: int = 16


@dataclass
class ServeConfig:
    name: str
    model: str
    engine: str = "vllm"
    group: str = ""
    label: str = ""
    revision: str | None = None
    quantization: str | None = None
    dtype: str = "bfloat16"
    max_num_seqs: int = 64
    enable_prefix_caching: bool = True
    gpu_memory_utilization: float = 0.90
    max_model_len: int | str = "auto"
    seed: int = 0
    system: str = "full"
    workloads: list[str] = field(default_factory=list)
    concurrency: list[int] = field(default_factory=list)
    rates: list[float] = field(default_factory=list)
    duration_s: float = 60.0
    warmup: int = 8
    max_requests_per_level: int = 3000
    quality: bool = False
    quality_guided: bool = False
    quality_approach: str | None = None
    lora: LoraSpec | None = None
    gguf_file: str | None = None
    n_gpu_layers: int = 99
    parallel: int = 16
    ctx_per_slot: int = 4096

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ServeConfig:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"config {d.get('name')!r}: unknown keys {sorted(unknown)}")
        d = dict(d)
        if isinstance(d.get("lora"), dict):
            d["lora"] = LoraSpec(**d["lora"])
        cfg = cls(**d)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.engine not in ENGINES:
            raise ValueError(f"{self.name}: engine must be one of {ENGINES}")
        bad = [w for w in self.workloads if w not in WORKLOADS]
        if bad:
            raise ValueError(f"{self.name}: unknown workloads {bad}")
        if not self.workloads or not (self.concurrency or self.rates):
            raise ValueError(f"{self.name}: needs workloads and at least one concurrency level or rate")
        if self.engine == "llama.cpp" and not self.gguf_file:
            raise ValueError(f"{self.name}: llama.cpp needs gguf_file")
        if self.system not in PROMPT_BUDGET:
            raise ValueError(f"{self.name}: system must be one of {sorted(PROMPT_BUDGET)}")

    @property
    def request_model(self) -> str:
        """The ``model`` field of each request (the LoRA adapter's name for the LoRA row)."""
        return self.lora.name if self.lora else self.model

    @property
    def n_levels(self) -> int:
        return len(self.concurrency) + len(self.rates)

    @property
    def quality_shots(self) -> int:
        return 0 if self.system == "compact" else 3

    def quality_passes(self) -> list[tuple[str, str]]:
        """(approach prefix, runner response_format) per quality pass: free decoding, then guided JSON."""
        if not self.quality:
            return []
        prefix = self.quality_approach or f"vllm-{self.name}"
        out = [(prefix, "none")]
        if self.quality_guided:
            out.append((f"{prefix}-guided", "json_schema"))
        return out

    def quality_approach_ids(self) -> list[str]:
        suffix = "few_shot" if self.quality_shots else "zero_shot"
        return [f"{p}_{suffix}" for p, _ in self.quality_passes()]


def load_matrix(path: str | Path = DEFAULT_MATRIX) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    return data


def expand(matrix: dict[str, Any]) -> list[ServeConfig]:
    """Every config with defaults, load settings and estimate caps merged in (config keys win)."""
    defaults = dict(matrix.get("defaults") or {})
    load = {k: v for k, v in (matrix.get("load") or {}).items() if k != "sla"}
    cap = (matrix.get("estimate") or {}).get("max_requests_per_level")
    out = []
    names: set[str] = set()
    for c in matrix.get("configs") or []:
        merged = {**defaults, **load, **({"max_requests_per_level": cap} if cap else {}), **c}
        cfg = ServeConfig.from_dict(merged)
        if cfg.name in names:
            raise ValueError(f"duplicate config name {cfg.name!r}")
        names.add(cfg.name)
        out.append(cfg)
    return out


def select(configs: Sequence[ServeConfig], spec: str | Sequence[str]) -> list[ServeConfig]:
    """``'all'``, comma-separated names, or ``group:<name>`` entries; order follows the matrix."""
    items = [s.strip() for s in (spec.split(",") if isinstance(spec, str) else spec) if s.strip()]
    if not items:
        raise ValueError("no configuration selected (use 'all', names or group:<name>)")
    if "all" in items:
        return list(configs)
    by_name = {c.name: c for c in configs}
    chosen: set[str] = set()
    for it in items:
        if it.startswith("group:"):
            g = it.removeprefix("group:")
            members = [c.name for c in configs if c.group == g]
            if not members:
                raise ValueError(f"no config in group {g!r}")
            chosen.update(members)
        elif it in by_name:
            chosen.add(it)
        else:
            raise ValueError(f"unknown config {it!r}; known: {', '.join(by_name)}")
    return [c for c in configs if c.name in chosen]


def apply_overrides(configs: Sequence[ServeConfig], overrides: dict[str, Any] | None) -> list[ServeConfig]:
    """Request-file level overrides (concurrency, rates, duration_s, warmup, workloads) for every config."""
    allowed = {"concurrency", "rates", "duration_s", "warmup", "workloads", "max_requests_per_level"}
    overrides = overrides or {}
    bad = set(overrides) - allowed
    if bad:
        raise ValueError(f"unsupported level overrides {sorted(bad)}; allowed: {sorted(allowed)}")
    out = []
    for c in configs:
        d = c.to_dict()
        d.update(overrides)
        if "workloads" in overrides and c.system == "compact":
            d["workloads"] = [
                w if w.endswith("compact") else _compact_variant(w) for w in overrides["workloads"]
            ]
        out.append(ServeConfig.from_dict(d))
    return out


def _compact_variant(workload: str) -> str:
    return {"extract-full": "extract-compact", "synthetic-long": "synthetic-long-compact"}.get(
        workload, workload
    )


# ------------------------------------------------------------------------------------------- server flags


def model_len(cfg: ServeConfig) -> int:
    if cfg.max_model_len != "auto":
        return int(cfg.max_model_len)
    need = max(
        PROMPT_BUDGET[cfg.system] + OUTPUT_BUDGET[WORKLOADS[w].synthetic_output] for w in cfg.workloads
    )
    return int(math.ceil(need / 1024) * 1024)


def vllm_args(cfg: ServeConfig, *, port: int = 8000, lora_path: str | None = None) -> list[str]:
    """``vllm serve`` command line for one config."""
    args = [
        "vllm",
        "serve",
        cfg.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        cfg.model,
        "--dtype",
        cfg.dtype,
        "--max-model-len",
        str(model_len(cfg)),
        "--max-num-seqs",
        str(cfg.max_num_seqs),
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
        "--seed",
        str(cfg.seed),
        "--enable-prefix-caching" if cfg.enable_prefix_caching else "--no-enable-prefix-caching",
    ]
    if cfg.revision:
        args += ["--revision", cfg.revision]
    if cfg.quantization:
        args += ["--quantization", cfg.quantization]
    if cfg.lora:
        if not lora_path:
            raise ValueError(f"{cfg.name}: LoRA config needs the local adapter path")
        args += [
            "--enable-lora",
            "--lora-modules",
            f"{cfg.lora.name}={lora_path}",
            "--max-lora-rank",
            str(cfg.lora.max_lora_rank),
            "--max-loras",
            "1",
        ]
    return args


def llama_server_args(
    cfg: ServeConfig, gguf_path: str, *, port: int = 8000, binary: str = "/app/llama-server"
) -> list[str]:
    """``llama-server`` command line: all layers on the GPU, ``parallel`` slots, continuous batching."""
    return [
        binary,
        "-m",
        gguf_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--alias",
        cfg.request_model,
        "-ngl",
        str(cfg.n_gpu_layers),
        "--parallel",
        str(cfg.parallel),
        "-c",
        str(cfg.parallel * cfg.ctx_per_slot),
        "--cont-batching",
        "--metrics",
        "--seed",
        str(cfg.seed),
    ]


# ------------------------------------------------------------------------------------------- estimate


def hourly_rate(matrix: dict[str, Any], gpu: str | None = None) -> dict[str, float]:
    """USD per hour of one benchmark container: GPU + CPU cores + memory (Modal bills them separately)."""
    prices = matrix.get("prices") or {}
    gpu = gpu or str(matrix.get("gpu", "L4"))
    g = float(prices["gpu"][gpu]["usd_per_hour"])
    cont = matrix.get("container") or {}
    cpu = float(cont.get("cpu", 0)) * float((prices.get("cpu_core") or {}).get("usd_per_hour", 0.0))
    mem = float(cont.get("memory_gib", 0)) * float((prices.get("memory_gib") or {}).get("usd_per_hour", 0.0))
    return {"gpu": g, "cpu": round(cpu, 4), "memory": round(mem, 4), "total": round(g + cpu + mem, 4)}


def estimate(
    cfg: ServeConfig, matrix: dict[str, Any], *, skip_load: bool = False, skip_quality: bool = False
) -> dict[str, Any]:
    """GPU minutes and USD for one config: startup + workloads x levels x (duration + overhead) + quality."""
    est = matrix.get("estimate") or {}
    startup = float((est.get("startup_min") or {}).get(cfg.engine, 4.0))
    overhead = float(est.get("level_overhead_s", 20))
    load_min = 0.0 if skip_load else len(cfg.workloads) * cfg.n_levels * (cfg.duration_s + overhead) / 60.0
    quality_min = 0.0 if skip_quality else len(cfg.quality_passes()) * float(est.get("quality_pass_min", 3.0))
    minutes = startup + load_min + quality_min
    rate = hourly_rate(matrix)
    return {
        "config": cfg.name,
        "startup_min": round(startup, 1),
        "load_min": round(load_min, 1),
        "quality_min": round(quality_min, 1),
        "minutes": round(minutes, 1),
        "usd_gpu": round(minutes / 60 * rate["gpu"], 2),
        "usd_total": round(minutes / 60 * rate["total"], 2),
    }


def timeout_s(est_minutes: float) -> int:
    """Per-container timeout: 1.5 x estimate + 15 min (first weight download), capped at Modal's 24 h."""
    return int(min(24 * 3600, (est_minutes * 1.5 + 15) * 60))


@dataclass
class BudgetDecision:
    estimates: list[dict[str, Any]]
    total_minutes: float
    total_usd_gpu: float
    total_usd: float
    cap_minutes: float
    ok: bool
    message: str


def budget(
    configs: Sequence[ServeConfig],
    matrix: dict[str, Any],
    *,
    cap_minutes: float | None = None,
    yes: bool = False,
    skip_load: bool = False,
    skip_quality: bool = False,
) -> BudgetDecision:
    """Refuse (ok=False) when the estimate exceeds the cap, unless ``yes``."""
    cap = float(cap_minutes or matrix.get("max_gpu_minutes", 420))
    ests = [estimate(c, matrix, skip_load=skip_load, skip_quality=skip_quality) for c in configs]
    total = round(sum(e["minutes"] for e in ests), 1)
    usd_gpu = round(sum(e["usd_gpu"] for e in ests), 2)
    usd = round(sum(e["usd_total"] for e in ests), 2)
    over = total > cap
    if over and not yes:
        msg = (
            f"estimate {total} GPU-min exceeds the cap of {cap:g} (MAX_GPU_MINUTES); pass --yes to run anyway"
        )
    elif over:
        msg = f"estimate {total} GPU-min exceeds the cap of {cap:g}, running because --yes was given"
    else:
        msg = f"estimate {total} GPU-min within the cap of {cap:g}"
    return BudgetDecision(ests, total, usd_gpu, usd, cap, not over or yes, msg)


def format_estimate(d: BudgetDecision, gpu: str = "L4") -> str:
    lines = [
        f"| config | startup | load | quality | {gpu} min | USD (GPU) | USD (GPU+CPU+RAM) |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in d.estimates:
        lines.append(
            f"| {e['config']} | {e['startup_min']} | {e['load_min']} | {e['quality_min']} | {e['minutes']} | "
            f"{e['usd_gpu']:.2f} | {e['usd_total']:.2f} |"
        )
    lines.append(
        f"| **total** | | | | **{d.total_minutes}** ({d.total_minutes / 60:.1f} h) | **{d.total_usd_gpu:.2f}** | "
        f"**{d.total_usd:.2f}** |"
    )
    lines.append("")
    lines.append(d.message)
    return "\n".join(lines)


class GpuBudget:
    """Runtime guard: a config starts only if actual minutes used + estimates in flight + its own estimate stay
    within the cap (so a mis-estimate cannot snowball across the remaining configs)."""

    def __init__(self, cap_minutes: float) -> None:
        self.cap = cap_minutes
        self.used = 0.0
        self.reserved: dict[str, float] = {}

    def try_start(self, name: str, est_minutes: float) -> bool:
        if self.used + sum(self.reserved.values()) + est_minutes > self.cap:
            return False
        self.reserved[name] = est_minutes
        return True

    def finish(self, name: str, actual_minutes: float) -> None:
        self.reserved.pop(name, None)
        self.used += actual_minutes


# ------------------------------------------------------------------------------------------- request file


@dataclass
class BenchRequestFile:
    configs: list[str]
    overrides: dict[str, Any]
    yes: bool = False
    skip_load: bool = False
    skip_quality: bool = False
    max_gpu_minutes: float | None = None


def parse_request(data: dict[str, Any], configs: Sequence[ServeConfig]) -> BenchRequestFile:
    """Validate ``.github/bench-request.json``: {"configs": "all" | [...], "levels": {...}, "yes": bool, ...}."""
    allowed = {"configs", "levels", "yes", "skip_load", "skip_quality", "max_gpu_minutes", "note", "id"}
    bad = set(data) - allowed
    if bad:
        raise ValueError(f"unknown request keys {sorted(bad)}; allowed: {sorted(allowed)}")
    spec = data.get("configs", "all")
    chosen = select(configs, spec if isinstance(spec, str) else list(spec))
    overrides = dict(data.get("levels") or {})
    apply_overrides(chosen, overrides)  # validates
    cap = data.get("max_gpu_minutes")
    return BenchRequestFile(
        configs=[c.name for c in chosen],
        overrides=overrides,
        yes=bool(data.get("yes", False)),
        skip_load=bool(data.get("skip_load", False)),
        skip_quality=bool(data.get("skip_quality", False)),
        max_gpu_minutes=float(cap) if cap is not None else None,
    )


# ------------------------------------------------------------------------------------------- manifest


def build_manifest(
    cfg: ServeConfig,
    *,
    argv: Sequence[str],
    engine_version: str | None,
    gpu_name: str | None,
    model_revision: str | None,
    started_utc: str,
    wall_s: float,
    usd_per_hour: float,
    files: Sequence[str],
    estimate_minutes: float | None = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything needed to interpret (and repeat) one config's run."""
    return {
        "config": cfg.name,
        "label": cfg.label or cfg.name,
        "group": cfg.group,
        "engine": cfg.engine,
        "engine_version": engine_version,
        "model": cfg.model,
        "model_revision": model_revision,
        "request_model": cfg.request_model,
        "gpu": gpu_name,
        "argv": list(argv),
        "settings": cfg.to_dict(),
        "started_utc": started_utc,
        "wall_s": round(wall_s, 1),
        "gpu_minutes": round(wall_s / 60, 2),
        "estimate_minutes": estimate_minutes,
        "usd_per_hour": usd_per_hour,
        "estimated_cost_usd": round(wall_s / 3600 * usd_per_hour, 4),
        "files": list(files),
        "status": "error" if error else "ok",
        "error": error,
        **(extra or {}),
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    return p


# ------------------------------------------------------------------------------------------- scheduling


@dataclass
class Selection:
    configs: list[ServeConfig]
    decision: BudgetDecision
    request: BenchRequestFile | None = None
    skip_load: bool = False
    skip_quality: bool = False


def plan_selection(
    matrix: dict[str, Any],
    *,
    config: str = "",
    request: dict[str, Any] | None = None,
    levels: dict[str, Any] | None = None,
    yes: bool = False,
    cap_minutes: float | None = None,
    skip_load: bool = False,
    skip_quality: bool = False,
    quality_only: bool = False,
) -> Selection:
    """Resolve what to run from CLI arguments and/or a request file (CLI arguments win). ``quality_only``
    keeps only configs with a quality pass and skips their load test."""
    all_cfgs = expand(matrix)
    req = parse_request(request, all_cfgs) if request is not None else None
    spec = config or (",".join(req.configs) if req else "")
    chosen = select(all_cfgs, spec)
    overrides = {**(req.overrides if req else {}), **(levels or {})}
    chosen = apply_overrides(chosen, overrides)
    yes = yes or bool(req and req.yes)
    cap = cap_minutes or (req.max_gpu_minutes if req else None)
    skip_load = skip_load or quality_only or bool(req and req.skip_load)
    skip_quality = skip_quality or bool(req and req.skip_quality)
    if quality_only:
        chosen = [c for c in chosen if c.quality]
        if not chosen:
            raise ValueError("none of the selected configs has a quality pass")
    decision = budget(
        chosen, matrix, cap_minutes=cap, yes=yes, skip_load=skip_load, skip_quality=skip_quality
    )
    return Selection(chosen, decision, req, skip_load, skip_quality)


def run_scheduled(
    configs: Sequence[ServeConfig],
    estimates: dict[str, float],
    *,
    cap_minutes: float,
    max_parallel: int,
    submit: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run configs at most ``max_parallel`` at a time under a :class:`GpuBudget`. ``submit(cfg)`` returns a
    ``concurrent.futures.Future`` resolving to the run's manifest. A config that no longer fits the budget is
    skipped once nothing else is running. Returns (manifests, skipped names)."""
    from concurrent.futures import FIRST_COMPLETED, Future, wait

    guard = GpuBudget(cap_minutes)
    pending = list(configs)
    running: dict[Future[dict[str, Any]], ServeConfig] = {}
    done: list[dict[str, Any]] = []
    skipped: list[str] = []
    while pending or running:
        while pending and len(running) < max_parallel:
            cfg = pending[0]
            if not guard.try_start(cfg.name, estimates[cfg.name]):
                break
            pending.pop(0)
            running[submit(cfg)] = cfg
        if not running:
            skipped.append(pending.pop(0).name)  # does not fit even with nothing in flight
            continue
        finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
        for fut in finished:
            cfg = running.pop(fut)
            try:
                manifest = fut.result()
            except Exception as e:  # the container failed outright: charge the estimate
                manifest = {"config": cfg.name, "status": "error", "error": f"{type(e).__name__}: {e}"}
            guard.finish(cfg.name, float(manifest.get("gpu_minutes", estimates[cfg.name])))
            done.append(manifest)
    return done, skipped
