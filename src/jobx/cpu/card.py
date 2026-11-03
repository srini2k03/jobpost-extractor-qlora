"""MODEL_CARD.md for the CPU-route release (also used as the GitHub Release notes)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jobx import prompts
from jobx.cards import metrics_table
from jobx.cpu.train import load_route_config

APPROACHES = ("qwen2.5-0.5b-lora-cpu_f16", "qwen2.5-0.5b-lora-cpu_q8_0", "qwen2.5-0.5b-lora-cpu_q4_k_m")


def bench_table(root: Path) -> str:
    rows = []
    for p in sorted((root / "results" / "benchmarks").glob("cpu_lora_*.json")):
        rows += json.loads(p.read_text()).get("rows", [])
    if not rows:
        return "_pending: the evaluation workflow has not run yet_"
    out = [
        "| variant | size (GB) | micro F1 % | tokens/s | mean latency (s) | peak RAM (GB) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        f1 = r.get("micro_f1")
        out.append(
            f"| {r['variant']} | {r.get('size_gb')} | {'-' if f1 is None else f'{100 * f1:.1f}'} | "
            f"{r.get('tokens_per_s')} | {r.get('mean_latency_s')} | {r.get('peak_ram_gb')} |"
        )
    return "\n".join(out)


def card_values(root: Path) -> dict[str, Any]:
    cfg = load_route_config(root / "configs" / "cpu" / "route.yaml")
    plan_path = root / "results" / "training" / cfg["run_name"] / "plan.json"
    info = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    plan = info.get("plan", {})
    return {
        "lora_r": cfg["lora"]["r"],
        "lora_alpha": cfg["lora"]["alpha"],
        "learning_rate": cfg["training"]["learning_rate"],
        "llama_cpp_ref": cfg["llama_cpp_ref"],
        "system_prompt": prompts.system_prompt(cfg["system_prompt"]),
        "n_train": plan.get("n_train", "?"),
        "max_length": plan.get("max_length", "?"),
        "dropped": plan.get("dropped_too_long", "?"),
        "epochs": plan.get("epochs", "?"),
        "max_steps": plan.get("max_steps", "?"),
        "seconds_per_step": plan.get("seconds_per_step", "?"),
        "plan_reason": plan.get("reason", "pending"),
        "cpu": info.get("cpu", "?"),
        "threads": info.get("threads", "?"),
        "precision": "bf16" if info.get("bf16") else "fp32",
        "metrics_table": metrics_table(list(APPROACHES), root),
        "benchmark_table": bench_table(root),
    }


def render_model_card(root: Path) -> str:
    text = (root / "model_cards" / "cpu_release.md").read_text()
    vals = card_values(root)
    return re.sub(r"\{\{\s*([a-z_0-9]+)\s*\}\}", lambda m: str(vals.get(m.group(1), m.group(0))), text)


def write_model_card(root: str | Path = ".", out: str | Path = "MODEL_CARD.md") -> Path:
    root = Path(root)
    p = Path(out) if Path(out).is_absolute() else root / out
    p.write_text(render_model_card(root))
    return p
