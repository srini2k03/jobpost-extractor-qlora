"""Training config loading: YAML with optional ``inherit`` and dotted CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from jobx import REPO_ROOT


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() or p.exists() else REPO_ROOT / p


def load_train_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load(_resolve(path).read_text())
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = deep_merge(load_train_config(parent), cfg)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """["training.learning_rate=1e-4", "lora.r=8"] -> nested dict (values parsed as YAML)."""
    out: dict[str, Any] = {}
    for it in items:
        key, _, raw = it.partition("=")
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def sweep_runs(sweep_path: str | Path) -> list[tuple[str, dict[str, Any]]]:
    sw = yaml.safe_load(_resolve(sweep_path).read_text())
    base = load_train_config(sw["base"])
    runs = []
    for r in sw["runs"]:
        name = r["name"]
        over = deep_merge(sw.get("common_overrides", {}), {k: v for k, v in r.items() if k != "name"})
        cfg = deep_merge(base, over)
        cfg["run_name"] = f"sweep-{name}"
        cfg["training"]["output_dir"] = f"outputs/sweep-{name}"
        runs.append((name, cfg))
    return runs
