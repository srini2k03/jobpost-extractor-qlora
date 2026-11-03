"""Plot training curves from a Trainer ``log_history.json`` (exported by the training notebook)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def split_history(history: list[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = {
        "train_loss": [],
        "eval_loss": [],
        "eval_json_f1": [],
        "learning_rate": [],
    }
    for row in history:
        step = float(row.get("step", 0))
        if "loss" in row:
            series["train_loss"].append((step, float(row["loss"])))
        if "eval_loss" in row:
            series["eval_loss"].append((step, float(row["eval_loss"])))
        if "eval_json_f1" in row:
            series["eval_json_f1"].append((step, float(row["eval_json_f1"])))
        if "learning_rate" in row:
            series["learning_rate"].append((step, float(row["learning_rate"])))
    return series


def plot_curves(log_history: str | Path, out_png: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = json.loads(Path(log_history).read_text())
    s = split_history(hist)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for key, ax_i in (("train_loss", 0), ("eval_loss", 0)):
        if s[key]:
            xs, ys = zip(*s[key], strict=True)
            axes[ax_i].plot(xs, ys, label=key.replace("_", " "))
    axes[0].set_title("loss")
    axes[0].set_xlabel("step")
    axes[0].legend()
    if s["eval_json_f1"]:
        xs, ys = zip(*s["eval_json_f1"], strict=True)
        axes[1].plot(xs, ys, marker="o")
    axes[1].set_title("validation JSON micro F1")
    axes[1].set_xlabel("step")
    if s["learning_rate"]:
        xs, ys = zip(*s["learning_rate"], strict=True)
        axes[2].plot(xs, ys)
    axes[2].set_title("learning rate")
    axes[2].set_xlabel("step")
    fig.tight_layout()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
