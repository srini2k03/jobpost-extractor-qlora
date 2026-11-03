"""`make local-check`: is this machine ready for the GPU steps? Prints a PASS/WARN/FAIL table.

Every check is independent and never raises, so a machine without a GPU (or without torch) still gets
a full report. Exit code 1 if any check FAILs.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import re
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from jobx import REPO_ROOT

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
MIN_FREE_DISK_GB = 60.0  # 7B teacher (~15 GB) + 1.5B base/merged/AWQ (~8 GB) + checkpoints + pip caches
MIN_VRAM_GB = 10.0


@dataclass
class Check:
    name: str
    status: str
    detail: str


def pinned_versions(req_file: Path) -> dict[str, str]:
    """name -> version for every `name==version` line of a requirements file."""
    out: dict[str, str] = {}
    for line in req_file.read_text().splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)==([^\s;#]+)", line)
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def _installed(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def check_python() -> Check:
    v = sys.version_info
    ok = (3, 11) <= (v.major, v.minor) < (3, 13)
    return Check("python", PASS if ok else WARN, f"{platform.python_version()} (3.11 or 3.12 recommended)")


def check_platform() -> Check:
    rel = platform.release().lower()
    wsl = "microsoft" in rel or "wsl" in rel
    system = platform.system()
    if system == "Windows":
        return Check("platform", WARN, "native Windows: run inside WSL2 Ubuntu (see docs/local_gpu_setup.md)")
    return Check("platform", PASS, f"{system} {platform.release()}" + (" (WSL2)" if wsl else ""))


def check_versions(req_file: Path) -> Check:
    pins = pinned_versions(req_file)
    bad = []
    for name, want in pins.items():
        have = _installed(name)
        base = (have or "").split("+")[0]  # torch reports e.g. 2.8.0+cu128
        if base != want:
            bad.append(f"{name} {have or 'missing'} (want {want})")
    if not bad:
        return Check("package versions", PASS, f"{len(pins)} pins match {req_file.name}")
    return Check("package versions", FAIL, "; ".join(bad[:6]) + (" ..." if len(bad) > 6 else ""))


def check_torch_cuda() -> list[Check]:
    try:
        import torch
    except ImportError:
        return [
            Check("torch", FAIL, "not installed (uv pip install -r requirements/local-gpu.txt)"),
            Check("cuda", FAIL, "needs torch"),
            Check("bf16", FAIL, "needs torch"),
        ]
    out = [Check("torch", PASS, f"{torch.__version__}, CUDA build {torch.version.cuda}")]
    if not torch.cuda.is_available():
        return [
            *out,
            Check(
                "cuda",
                FAIL,
                "torch.cuda.is_available() is False: driver/WSL GPU not visible (run nvidia-smi inside WSL)",
            ),
            Check("bf16", FAIL, "needs CUDA"),
        ]
    p = torch.cuda.get_device_properties(0)
    vram = p.total_memory / 1e9
    out.append(
        Check(
            "cuda",
            PASS if vram >= MIN_VRAM_GB else WARN,
            f"{p.name}, {vram:.1f} GB VRAM, compute {p.major}.{p.minor}",
        )
    )
    bf16 = torch.cuda.is_bf16_supported()
    out.append(
        Check(
            "bf16",
            PASS if bf16 else WARN,
            "supported" if bf16 else "not supported: profile falls back to fp16",
        )
    )
    return out


def check_bnb_4bit() -> Check:
    """Load a tiny randomly initialised Llama in 4-bit NF4 (built locally: no download) and run a forward pass."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaConfig
    except ImportError as e:
        return Check("bitsandbytes 4-bit", FAIL, f"import failed: {e}")
    if not torch.cuda.is_available():
        return Check("bitsandbytes 4-bit", FAIL, "needs CUDA")
    try:
        import bitsandbytes  # noqa: F401

        cfg = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=256,
        )
        with tempfile.TemporaryDirectory() as d:
            AutoModelForCausalLM.from_config(cfg).save_pretrained(d)
            q = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
            m = AutoModelForCausalLM.from_pretrained(d, quantization_config=q, device_map={"": 0})
            with torch.no_grad():
                m(torch.randint(0, 256, (1, 8), device="cuda"))
        return Check(
            "bitsandbytes 4-bit", PASS, f"bitsandbytes {_installed('bitsandbytes')}: NF4 load + forward OK"
        )
    except Exception as e:
        return Check("bitsandbytes 4-bit", FAIL, f"{type(e).__name__}: {str(e)[:160]}")


def check_disk(path: Path) -> Check:
    free = shutil.disk_usage(path).free / 1e9
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    st = PASS if free >= MIN_FREE_DISK_GB else (WARN if free >= 30 else FAIL)
    return Check(
        "free disk", st, f"{free:.0f} GB free at {path} (want >= {MIN_FREE_DISK_GB:.0f}); HF cache: {hf_home}"
    )


def check_ram() -> Check:
    try:
        kb = next(
            int(x.split()[1])
            for x in Path("/proc/meminfo").read_text().splitlines()
            if x.startswith("MemTotal")
        )
    except (OSError, StopIteration, ValueError):
        return Check("system RAM", WARN, "could not read /proc/meminfo")
    gb = kb / 1e6
    st = PASS if gb >= 20 else WARN
    return Check(
        "system RAM",
        st,
        f"{gb:.0f} GB visible (WSL default is half of Windows RAM; set ~24 GB in .wslconfig)",
    )


def check_hf() -> list[Check]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except Exception:
            token = None
    out = [
        Check(
            "HF login",
            PASS if token else FAIL,
            "token found" if token else "run `huggingface-cli login` (write token) or export HF_TOKEN",
        )
    ]
    try:
        from jobx.hub import hf_username

        out.append(Check("HF_USERNAME", PASS, hf_username()))
    except Exception as e:
        out.append(Check("HF_USERNAME", FAIL, str(e)[:160]))
    return out


def check_wandb() -> Check:
    if os.environ.get("WANDB_API_KEY"):
        return Check("W&B login", PASS, "WANDB_API_KEY set")
    netrc = Path.home() / ".netrc"
    if netrc.exists() and "api.wandb.ai" in netrc.read_text(errors="ignore"):
        return Check("W&B login", PASS, "~/.netrc has api.wandb.ai")
    return Check("W&B login", WARN, "not logged in: training runs without W&B (`wandb login` to enable)")


def run_checks(
    req_file: Path | None = None, *, bnb: bool = True, extra: list[Callable[[], Check]] | None = None
) -> list[Check]:
    req_file = req_file or REPO_ROOT / "requirements" / "local-gpu.txt"
    checks: list[Check] = [check_python(), check_platform(), check_versions(req_file)]
    checks += check_torch_cuda()
    if bnb:
        checks.append(check_bnb_4bit())
    checks += [check_disk(REPO_ROOT), check_ram(), *check_hf(), check_wandb()]
    for fn in extra or []:
        checks.append(fn())
    return checks


def render(checks: list[Check]) -> str:
    w = max(len(c.name) for c in checks)
    lines = [f"{'check'.ljust(w)}  status  detail", f"{'-' * w}  ------  ------"]
    lines += [f"{c.name.ljust(w)}  {c.status:<6}  {c.detail}" for c in checks]
    n_fail = sum(c.status == FAIL for c in checks)
    n_warn = sum(c.status == WARN for c in checks)
    lines.append("")
    lines.append("RESULT: " + ("FAIL" if n_fail else "PASS") + f" ({n_fail} failed, {n_warn} warnings)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--requirements", default=str(REPO_ROOT / "requirements" / "local-gpu.txt"))
    ap.add_argument("--no-bnb", action="store_true", help="skip the 4-bit load test")
    a = ap.parse_args(argv)
    checks = run_checks(Path(a.requirements), bnb=not a.no_bnb)
    print(render(checks))
    return 1 if any(c.status == FAIL for c in checks) else 0
