"""Evaluate prediction files.

Prediction files live in ``results/predictions/<approach>__<split>.jsonl``; each line is
``{"id": ..., "output": "<raw model text>", "latency_s": optional, ...}``. A sidecar
``<same stem>.meta.json`` records approach, model, decoding, hardware, date and commit.
"""

from __future__ import annotations

import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jobx.data.build import read_jsonl
from jobx.eval.metrics import ExampleScore, aggregate, score_example

SPLIT_FILES = {
    "train": "data/processed/train.jsonl",
    "val": "data/processed/val.jsonl",
    "test_synth": "data/processed/test_synth.jsonl",
    "test_real": "data/real/test_real.jsonl",
}


def eval_subset(examples: list[dict[str, Any]], n: int | None) -> list[dict[str, Any]]:
    """Deterministic, evenly spaced subset by id (same subset for every approach)."""
    ex = sorted(examples, key=lambda e: e["id"])
    if not n or n >= len(ex):
        return ex
    step = len(ex) / n
    return [ex[int(i * step)] for i in range(n)]


def parse_pred_name(path: str | Path) -> tuple[str, str]:
    stem = Path(path).name.removesuffix(".jsonl")
    approach, _, split = stem.rpartition("__")
    if not approach:
        raise ValueError(f"prediction file must be named <approach>__<split>.jsonl, got {path}")
    return approach, split


def load_gold(split: str, root: str | Path = ".") -> dict[str, dict[str, Any]]:
    path = Path(root) / SPLIT_FILES[split]
    return {r["id"]: r for r in read_jsonl(path)}


def score_file(pred_path: str | Path, root: str | Path = ".") -> tuple[list[ExampleScore], dict[str, Any]]:
    _, split = parse_pred_name(pred_path)
    gold = load_gold(split, root)
    preds = read_jsonl(pred_path)
    scores = []
    missing = 0
    for p in preds:
        g = gold.get(p["id"])
        if g is None:
            missing += 1
            continue
        scores.append(score_example(p["id"], g["label"], p.get("output") or ""))
    lat = [float(p["latency_s"]) for p in preds if p.get("latency_s") is not None]
    info = {
        "split": split,
        "n_gold": len(gold),
        "n_pred": len(preds),
        "unknown_ids": missing,
        "coverage": round(len(scores) / len(gold), 4) if gold else 0.0,
        "mean_latency_s": round(sum(lat) / len(lat), 4) if lat else None,
    }
    return scores, info


def _clean(o: Any) -> Any:
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    return o


def git_commit(root: str | Path = ".") -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, check=True
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def evaluate_file(pred_path: str | Path, root: str | Path = ".", *, n_boot: int = 1000) -> dict[str, Any]:
    approach, _ = parse_pred_name(pred_path)
    scores, info = score_file(pred_path, root)
    meta_path = Path(str(pred_path).removesuffix(".jsonl") + ".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    metrics = aggregate(scores, n_boot=n_boot)
    return _clean(
        {
            "approach": approach,
            **info,
            "meta": meta,
            "metrics": metrics,
            "evaluated_at": datetime.now(UTC).strftime("%Y-%m-%d"),
        }
    )


def evaluate_all(
    pred_dir: str | Path, out_dir: str | Path, root: str | Path = ".", *, n_boot: int = 1000
) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for p in sorted(Path(pred_dir).glob("*__*.jsonl")):
        _, split = parse_pred_name(p)
        if not (Path(root) / SPLIT_FILES.get(split, "__missing__")).exists():
            continue
        res = evaluate_file(p, root, n_boot=n_boot)
        res.pop("evaluated_at")  # keep files byte-stable so CI can diff them
        dest = out / (p.name.removesuffix(".jsonl") + ".json")
        dest.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
        written.append(dest)
    return written


def write_metrics(pred_path: str | Path, root: str | Path = ".", *, n_boot: int = 1000) -> dict[str, Any]:
    """Score one predictions file into results/metrics/<stem>.json; returns the metrics dict."""
    res = evaluate_file(pred_path, root, n_boot=n_boot)
    res.pop("evaluated_at", None)
    dest = Path(root) / "results" / "metrics" / (Path(pred_path).name.removesuffix(".jsonl") + ".json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    m: dict[str, Any] = res["metrics"]
    return m
