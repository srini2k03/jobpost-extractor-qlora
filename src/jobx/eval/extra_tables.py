"""Markdown tables for the robustness suite, calibration, quantisation sweep and LoRA study.

Appended to results/results_table.md (and the README) by ``jobx results-table``; each table appears only when
its committed source file exists, so the README never shows a number that is not in the repository.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jobx.eval.dashboard import ROB_LABEL, SHORT


def _p(v: Any) -> str:
    return "-" if v is None else f"{100 * float(v):.1f}"


def _read(p: Path) -> Any:
    return json.loads(p.read_text()) if p.exists() else None


def robustness_md(root: Path) -> list[str]:
    rep = _read(root / "results/robustness/summary.json")
    if not rep:
        return []
    order = sorted(rep, key=lambda a: (list(SHORT).index(a) if a in SHORT else 99, a))
    md = [
        "#### Robustness suite (`robustness`, 14 categories x 12 generated cases)",
        "",
        '_Micro-F1 % per category (gold by construction). "Not a job post": share of outputs with every field empty. '
        "Injection success: share of prompt-injection cases where an injected value reached the output (lower is better)._",
        "",
        "| Category | " + " | ".join(SHORT.get(a, a) for a in order) + " |",
        "|---|" + "---|" * len(order),
    ]
    for cat, label in ROB_LABEL.items():
        key = "all_null_rate" if cat == "irrelevant" else "micro_f1"
        md.append(f"| {label} | " + " | ".join(_p((rep[a].get(cat) or {}).get(key)) for a in order) + " |")
    md.append(
        "| **Injection success %** | "
        + " | ".join(_p((rep[a].get("prompt_injection") or {}).get("attack_success_rate")) for a in order)
        + " |"
    )
    md.append("")
    return md


def calibration_md(root: Path) -> list[str]:
    out: list[str] = []
    for p in sorted((root / "results/calibration").glob("*.json")):
        if p.name == "calibrator.json":
            continue
        rep = json.loads(p.read_text())
        out += [
            f"#### Field-confidence calibration (`{rep['approach']}`)",
            "",
            f"_Fitted on val ({rep['n_fields_val']} fields); abstention threshold {rep['threshold']:.2f} = smallest with "
            f"{100 * rep['target_selective_accuracy']:.0f}% selective accuracy on val. Held-out splits not used for fitting._",
            "",
            "| Split | Fields | ECE raw | ECE calibrated | Brier calibrated | AUROC | Kept at threshold % | Accuracy kept % | Accuracy all % |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for split, s in rep["splits"].items():
            raw, cal, at = s["raw_token_probability"], s["calibrated"], s["at_threshold"]
            out.append(
                f"| {split} | {s['n_fields']} | {raw['ece']:.3f} | {cal['ece']:.3f} | {cal['brier']:.3f} | "
                f"{cal['auroc'] if cal['auroc'] is not None else '-'} | {_p(at['coverage'])} | "
                f"{_p(at['selective_accuracy'])} | {_p(at['accuracy_all_fields'])} |"
            )
        out.append("")
    return out


def quant_md(root: Path) -> list[str]:
    sw = _read(root / "results/quant/sweep.json")
    if not sw or not sw.get("variants"):
        return []
    sub = sw.get("subsets", {})
    md = [
        "#### Quantisation sweep (one runner, all variants)",
        "",
        f"_{sw.get('hardware', '')}, llama.cpp {sw.get('llama_cpp') or ''}, measured {sw.get('date', '')} @{sw.get('commit', '')}; "
        f"accuracy on {sub.get('test_synth', 0)} evenly spaced test_synth and {sub.get('test_real', 0)} test_real postings._",
        "",
        "| GGUF | Size (GB) | Synth micro-F1 % | Real micro-F1 % | JSON valid % (real) | Gen tokens/s | Prompt tokens/s | p50 latency (s) | p95 latency (s) | Cold start (s) | Peak RAM (GB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for v in sw["variants"]:
        acc = v.get("accuracy", {})
        lat = v.get("warm_latency_s") or {}
        md.append(
            f"| {v['quant']} | {v['size_gb']} | {_p((acc.get('test_synth') or {}).get('micro_f1'))} | "
            f"{_p((acc.get('test_real') or {}).get('micro_f1'))} | {_p((acc.get('test_real') or {}).get('json_validity'))} | "
            f"{v.get('gen_tokens_per_s') or '-'} | {v.get('prompt_tokens_per_s') or '-'} | {lat.get('p50') or '-'} | "
            f"{lat.get('p95') or '-'} | {v.get('startup_s', '-')} | {v.get('peak_ram_gb') or '-'} |"
        )
    if sw.get("thread_sweep"):
        md += [
            "",
            "| Threads ("
            + sw["thread_sweep"][0]["quant"]
            + ") | Gen tokens/s | Prompt tokens/s | p50 latency (s) |",
            "|---|---|---|---|",
        ]
        for t in sw["thread_sweep"]:
            md.append(
                f"| {t['threads']} | {t.get('gen_tokens_per_s') or '-'} | {t.get('prompt_tokens_per_s') or '-'} | "
                f"{(t.get('warm_latency_s') or {}).get('p50') or '-'} |"
            )
    md.append("")
    return md


def _cpu(name: str) -> str:
    import re

    m = re.search(r"(Xeon\(R\) [\w-]+|EPYC \w+)", name)
    return m.group(1).replace("(R)", "") if m else name


def lora_md(root: Path) -> list[str]:
    s = _read(root / "results/lora_study/summary.json")
    if not s or not s.get("runs"):
        return []
    md = [
        "#### LoRA study (reduced budget, validation only)",
        "",
        f"_{s.get('budget_note', '')} Selected: `{s['best']}` (within its 95% CI: {', '.join(s['within_best_ci']) or 'none'}). "
        "Seconds per step depend on the runner's CPU (listed); compare speed only between rows on the same CPU._",
        "",
        "| Config | r / alpha | Targets | LR | Trainable params | % of total | Adapter (MB) | s/step | CPU (precision) | Peak RAM (GB) | Val loss | Val micro-F1 % [95% CI] | JSON valid % | Hallucination % |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in s["runs"]:
        c, v = r["config"], r["val"]
        tgt = (
            "all linear"
            if len(c["target_modules"]) == 7
            else "attention"
            if len(c["target_modules"]) == 4
            else ",".join(c["target_modules"])
        )
        ci = v.get("ci95_micro_f1") or [None, None]
        md.append(
            f"| {r['id']} | {c['r']} / {c['alpha']} | {tgt} | {c['learning_rate']} | {r['params_trainable']:,} | "
            f"{100 * r['trainable_share']:.2f} | {r['adapter_mb']} | {r['seconds_per_step']} | "
            f"{_cpu(r.get('cpu', '?'))} ({'bf16' if r.get('bf16') else 'fp32'}) | {r['peak_rss_gb']} | "
            f"{r['val_loss']} | {_p(v['micro_f1'])} [{_p(ci[0])}, {_p(ci[1])}] | {_p(v['json_validity'])} | {_p(v['hallucination_rate'])} |"
        )
    md.append("")
    return md


def extra_md(root: Path) -> list[str]:
    return robustness_md(root) + calibration_md(root) + quant_md(root) + lora_md(root)
