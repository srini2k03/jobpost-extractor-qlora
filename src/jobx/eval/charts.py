"""Charts for the README and the Pages site, drawn from results/results.json (matplotlib, headless)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CHART_FILES = ("micro_f1_by_approach.png", "quant_tradeoff.png")


def _colors() -> dict[str, str]:
    return {"bar": "#4c78a8", "ours": "#e45756", "grid": "#d0d4d9", "text": "#333333"}


def f1_rows(data: dict[str, Any], split: str) -> list[tuple[str, float, tuple[float, float] | None]]:
    out = []
    for r in data.get("splits", {}).get(split, []):
        if r.get("status") != "measured":
            continue
        m = r["metrics"]
        ci = (r.get("ci95") or {}).get("micro_f1")
        out.append(
            (
                r["name"],
                float(m["micro_f1"]) * 100,
                (ci[0] * 100, ci[1] * 100) if ci and ci[0] is not None else None,
            )
        )
    return sorted(out, key=lambda x: x[1])


def draw_charts(root: str | Path = ".", out_dir: str | Path = "results/charts") -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, out = Path(root), Path(out_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((root / "results" / "results.json").read_text())
    c = _colors()
    written: list[Path] = []
    split = "test_real" if f1_rows(data, "test_real") else "test_synth"
    rows = f1_rows(data, split)
    if rows:
        fig, ax = plt.subplots(figsize=(9, 0.45 * len(rows) + 1.2))
        names = [n for n, _, _ in rows]
        vals = [v for _, v, _ in rows]
        colors = [c["ours"] if "fine-tune" in n.lower() else c["bar"] for n in names]
        ax.barh(names, vals, color=colors)
        for i, (_, v, ci) in enumerate(rows):
            if ci:
                ax.plot([ci[0], ci[1]], [i, i], color=c["text"], linewidth=1)
            ax.text(min(v + 1, 99), i, f"{v:.1f}", va="center", fontsize=8, color=c["text"])
        ax.set_xlim(0, 100)
        ax.set_xlabel(f"micro F1 % on {split} (95% bootstrap CI)")
        ax.grid(axis="x", color=c["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
        fig.tight_layout()
        p = out / CHART_FILES[0]
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)
    bench = [b for b in data.get("benchmarks", []) if b.get("size_gb") and b.get("micro_f1") is not None]
    if bench:
        fig, ax = plt.subplots(figsize=(7, 4))
        for b in bench:
            ax.scatter(
                b["size_gb"],
                b["micro_f1"] * 100,
                s=40 + 4 * float(b.get("tokens_per_s") or 0),
                color=c["bar"],
            )
            ax.annotate(
                f"{b['variant']}\n{b.get('tokens_per_s', '?')} tok/s",
                (b["size_gb"], b["micro_f1"] * 100),
                fontsize=7,
                xytext=(5, 5),
                textcoords="offset points",
            )
        ax.set_xlabel("size on disk (GB)")
        ax.set_ylabel("micro F1 %")
        ax.set_title("Size / accuracy / speed (marker size = tokens/s)", fontsize=9)
        ax.grid(color=c["grid"], linewidth=0.5)
        fig.tight_layout()
        p = out / CHART_FILES[1]
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)
    return written
