"""Background GPU memory/utilisation sampler.

Backends, in order: NVML via ``pynvml`` (package ``nvidia-ml-py``), then ``nvidia-smi --query-gpu``, then a
no-op. On a machine without a GPU (CI, laptops) the sampler records nothing and ``stats()`` reports
``backend='none'``, so the harness runs unchanged everywhere.

The sampler measures the GPU of the machine it runs on: run the load generator next to the server (in the
same Modal container in phase 2) for the numbers to mean anything. vLLM pre-allocates
``--gpu-memory-utilization`` of the card for its KV cache, so peak memory mostly reflects that setting; the
model-weight footprint comes from vLLM's own startup log.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from collections.abc import Callable
from types import TracebackType
from typing import Any

# One sample: (memory used MiB, memory total MiB, utilisation %), summed/averaged over visible GPUs.
Sample = tuple[float, float, float]
QueryFn = Callable[[], Sample | None]


def _nvml_query() -> QueryFn | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    except Exception:
        return None
    if not handles:
        return None

    def query() -> Sample | None:
        used = total = util = 0.0
        for h in handles:
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            used += mem.used / 2**20
            total += mem.total / 2**20
            util += pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        return used, total, util / len(handles)

    return query


def parse_nvidia_smi(text: str) -> Sample | None:
    """Parse ``--query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits``."""
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            rows.append(tuple(float(p) for p in parts))
        except ValueError:
            continue
    if not rows:
        return None
    return sum(r[0] for r in rows), sum(r[1] for r in rows), sum(r[2] for r in rows) / len(rows)


def _smi_query() -> QueryFn | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    cmd = [exe, "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]

    def query() -> Sample | None:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        return parse_nvidia_smi(out)

    return query if query() is not None else None


def detect_backend() -> tuple[str, QueryFn | None]:
    for name, factory in (("nvml", _nvml_query), ("nvidia-smi", _smi_query)):
        q = factory()
        if q is not None:
            return name, q
    return "none", None


class GpuSampler:
    """Context manager that polls the GPU every ``interval_s`` in a daemon thread."""

    def __init__(
        self,
        interval_s: float = 0.5,
        query: QueryFn | None = None,
        backend: str | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        if not enabled:
            backend, query = "none", None
        elif query is None:
            backend, query = detect_backend()
        self.backend = backend or "custom"
        self.query = query
        self.interval_s = interval_s
        self.samples: list[Sample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        assert self.query is not None
        while True:
            s = self.query()
            if s is not None:
                self.samples.append(s)
            if self._stop.wait(self.interval_s):
                return

    def start(self) -> GpuSampler:
        if self.query is not None and self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="gpu-sampler", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=10)
            self._thread = None
        return self.stats()

    def __enter__(self) -> GpuSampler:
        return self.start()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.stop()

    def stats(self) -> dict[str, Any]:
        if not self.samples:
            return {"backend": self.backend if self.query else "none", "n_samples": 0}
        return {
            "backend": self.backend,
            "n_samples": len(self.samples),
            "mem_peak_mib": round(max(s[0] for s in self.samples), 1),
            "mem_total_mib": round(max(s[1] for s in self.samples), 1),
            "util_mean": round(sum(s[2] for s in self.samples) / len(self.samples), 1),
            "util_max": round(max(s[2] for s in self.samples), 1),
        }
