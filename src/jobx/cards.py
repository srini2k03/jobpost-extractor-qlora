"""Fill model-card templates (model_cards/*.md) with repo names and measured metrics."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jobx import REPO_ROOT

PENDING = "_pending: no measured result yet_"


def metrics_table(approach_ids: list[str], root: Path = REPO_ROOT) -> str:
    rows = []
    for split in ("test_real", "test_synth"):
        for a in approach_ids:
            p = root / "results" / "metrics" / f"{a}__{split}.json"
            if not p.exists():
                continue
            m = json.loads(p.read_text())["metrics"]
            ci = (m.get("ci95") or {}).get("micro_f1") or [None, None]
            rows.append(
                f"| {a} | {split} | {m['n']} | {100 * m['micro_f1']:.1f}"
                + (f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if ci[0] is not None else "")
                + f" | {100 * m['json_validity']:.1f} | {100 * m['hallucination_rate']:.1f} |"
            )
    if not rows:
        return PENDING
    return "\n".join(
        [
            "| approach | split | n | micro F1 % [95% CI] | JSON valid % | hallucination % |",
            "|---|---|---|---|---|---|",
            *rows,
        ]
    )


def render_card(kind: str, values: dict[str, Any], root: Path = REPO_ROOT) -> str:
    text = (root / "model_cards" / f"{kind}.md").read_text()

    def sub(m: re.Match[str]) -> str:
        key = m.group(1)
        return str(values.get(key, m.group(0)))

    return re.sub(r"\{\{\s*([a-z_]+)\s*\}\}", sub, text)
