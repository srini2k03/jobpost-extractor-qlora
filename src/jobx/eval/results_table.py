"""Build the results tables (markdown + JSON) from results/metrics and results/benchmarks.

Every planned approach (configs/approaches.yaml) gets a row; rows without a metrics file
say "pending GPU run". Nothing here invents numbers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

PENDING = "pending GPU run"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _pct(v: Any) -> str:
    return "-" if v is None else f"{100 * float(v):.1f}"


def _ci(m: dict[str, Any], key: str) -> str:
    ci = (m.get("ci95") or {}).get(key)
    if not ci or ci[0] is None:
        return _pct(m.get(key))
    return f"{_pct(m.get(key))} [{_pct(ci[0])}, {_pct(ci[1])}]"


def load_metrics(metrics_dir: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for p in sorted(Path(metrics_dir).glob("*__*.json")):
        d = json.loads(p.read_text())
        out[(d["approach"], d["split"])] = d
    return out


def build_tables(root: str | Path = ".") -> tuple[str, dict[str, Any]]:
    root = Path(root)
    cfg = yaml.safe_load((root / "configs/approaches.yaml").read_text())
    metrics = load_metrics(root / "results/metrics")
    known = {a["id"] for a in cfg["approaches"]}
    approaches = list(cfg["approaches"]) + [
        {"id": a, "name": a, "model": "?", "decoding": "?", "hardware": "?", "produced_by": "?"}
        for a in sorted({k[0] for k in metrics} - known)
    ]
    md: list[str] = []
    data: dict[str, Any] = {"splits": {}, "benchmarks": []}
    split_titles = {
        "test_real": "Real postings (`test_real`, headline)",
        "test_synth": "Synthetic held-out families and companies (`test_synth`)",
    }
    for split in cfg["splits"]:
        md.append(f"#### {split_titles.get(split, split)}")
        md.append("")
        has_any = any(k[1] == split for k in metrics)
        if split == "test_real" and not has_any:
            md.append(
                "_The real test set is pending: it needs the fetch + draft-label notebook and a human "
                "pass with `jobx verify` (see the Colab checklist below)._"
            )
            md.append("")
        md.append(
            "| Approach | Hardware | n | JSON valid % | Schema valid % | Field EM % | Micro F1 % [95% CI] "
            "| Macro F1 % | Hallucination % [95% CI] | Status |"
        )
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        rows = []
        for a in approaches:
            d = metrics.get((a["id"], split))
            if d is None:
                md.append(
                    f"| {a['name']} | {a['hardware']} | - | - | - | - | - | - | - | {PENDING} |"
                    if a["hardware"] != "CPU"
                    else f"| {a['name']} | {a['hardware']} | - | - | - | - | - | - | - | pending (needs this split) |"
                )
                rows.append({"approach": a["id"], "name": a["name"], "status": "pending"})
                continue
            m = d["metrics"]
            date = (d.get("meta") or {}).get("date", "")
            commit = (d.get("meta") or {}).get("commit", "")
            status = "measured" + (f" {date}" if date else "") + (f" @{commit}" if commit else "")
            md.append(
                f"| {a['name']} | {a['hardware']} | {m['n']} | {_pct(m['json_validity'])} | "
                f"{_pct(m['schema_validity'])} | {_pct(m['field_em_mean'])} | {_ci(m, 'micro_f1')} | "
                f"{_pct(m['macro_f1'])} | {_ci(m, 'hallucination_rate')} | {status} |"
            )
            rows.append(
                {
                    "approach": a["id"],
                    "name": a["name"],
                    "status": "measured",
                    "n": m["n"],
                    "metrics": {
                        k: m.get(k)
                        for k in (
                            "json_validity",
                            "schema_validity",
                            "field_em_mean",
                            "micro_f1",
                            "macro_f1",
                            "hallucination_rate",
                            "record_exact_match",
                        )
                    },
                    "ci95": m.get("ci95"),
                    "per_field": m.get("per_field"),
                    "meta": d.get("meta"),
                }
            )
        data["splits"][split] = rows
        md.append("")
    # per-field table for every measured approach on the first split that has data
    for split in cfg["splits"]:
        measured = [(a, metrics[(a["id"], split)]) for a in approaches if (a["id"], split) in metrics]
        if not measured:
            continue
        fields = list(measured[0][1]["metrics"]["per_field"])
        md.append(f"#### Per-field exact match % (`{split}`)")
        md.append("")
        md.append("| Approach | " + " | ".join(fields) + " |")
        md.append("|---|" + "---|" * len(fields))
        for a, d in measured:
            pf = d["metrics"]["per_field"]
            md.append(f"| {a['name']} | " + " | ".join(_pct(pf[f]["em"]) for f in fields) + " |")
        md.append("")
    md += benchmark_table(root, data)
    return "\n".join(md).rstrip() + "\n", data


def benchmark_table(root: Path, data: dict[str, Any]) -> list[str]:
    planned = [
        ("fp16 (transformers)", "T4"),
        ("merged 8-bit (bitsandbytes)", "T4"),
        ("AWQ 4-bit", "T4"),
        ("GGUF Q8_0 (llama.cpp)", "CPU, GitHub Actions"),
        ("GGUF Q4_K_M (llama.cpp)", "CPU, GitHub Actions"),
    ]
    rows: dict[str, dict[str, Any]] = {}
    for p in sorted((root / "results/benchmarks").glob("*.json")):
        d = json.loads(p.read_text())
        for r in d.get("rows", []):
            rows[r["variant"]] = r
    md = [
        "#### Size / speed / accuracy trade-off",
        "",
        "| Variant | Hardware | Size (GB) | Micro F1 % | Tokens/s | Peak memory (GB) | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for variant, hw in planned:
        r = rows.get(variant)
        if r is None:
            md.append(
                f"| {variant} | {hw} | - | - | - | - | {PENDING if 'T4' in hw else 'pending (needs merged model)'} |"
            )
            continue
        mem = r.get("peak_ram_gb") or r.get("peak_vram_gb")
        md.append(
            f"| {variant} | {r.get('hardware', hw)} | {r.get('size_gb', '-')} | {_pct(r.get('micro_f1'))} | "
            f"{r.get('tokens_per_s', '-')} | {mem if mem is not None else '-'} | measured {r.get('date', '')} |"
        )
        data["benchmarks"].append(r)
    md.append("")
    return md


def update_readme(readme: str | Path, table_md: str) -> bool:
    p = Path(readme)
    text = p.read_text()
    if START not in text or END not in text:
        return False
    new = re.sub(
        re.escape(START) + r".*?" + re.escape(END), f"{START}\n{table_md}{END}", text, flags=re.DOTALL
    )
    if new != text:
        p.write_text(new)
    return True


def write_all(root: str | Path = ".") -> None:
    root = Path(root)
    md, data = build_tables(root)
    (root / "results/results_table.md").write_text(md)
    (root / "results/results.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    update_readme(root / "README.md", md)
