"""Evaluation dashboard sections of the Pages site, rendered from committed result files only.

Every function returns an HTML fragment (or "" when its source file does not exist yet, so the page never
shows an invented number). Sources: results/results.json, results/robustness/summary.json,
results/calibration/<approach>.json, results/quant/sweep.json, data/processed/stats.json,
data/real/real_meta.json, results/audit/leakage.json and the committed prediction files.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any

OURS = "qwen2.5-0.5b-lora-cpu_q4_k_m"
BASE = "qwen2.5-0.5b-cpu_zero_shot"
FIELD_SET = (
    "regex_baseline",
    "qwen2.5-0.5b-cpu_zero_shot",
    "qwen2.5-0.5b-cpu_few_shot_guided",
    "qwen2.5-1.5b-cpu_zero_shot",
    "qwen2.5-3b-cpu_zero_shot",
    "qwen2.5-0.5b-lora-cpu_f16",
    "qwen2.5-0.5b-lora-cpu_q4_k_m",
    "qwen2.5-0.5b-lora-cpu_q4_k_m_guided",
)
SHORT = {
    "regex_baseline": "Regex rules",
    "qwen2.5-0.5b-cpu_zero_shot": "0.5B zero-shot",
    "qwen2.5-0.5b-cpu_zero_shot_guided": "0.5B zero-shot + grammar",
    "qwen2.5-0.5b-cpu_few_shot": "0.5B 3-shot",
    "qwen2.5-0.5b-cpu_few_shot_guided": "0.5B 3-shot + grammar",
    "qwen2.5-1.5b-cpu_zero_shot": "1.5B zero-shot",
    "qwen2.5-1.5b-cpu_few_shot": "1.5B 3-shot",
    "qwen2.5-3b-cpu_zero_shot": "3B zero-shot",
    "qwen2.5-0.5b-lora-cpu_f16": "LoRA 0.5B F16",
    "qwen2.5-0.5b-lora-cpu_q8_0": "LoRA 0.5B Q8_0",
    "qwen2.5-0.5b-lora-cpu_q4_k_m": "LoRA 0.5B Q4_K_M",
    "qwen2.5-0.5b-lora-cpu_q4_k_m_guided": "LoRA Q4_K_M + grammar",
    "gpt-oss-120b-groq_few_shot": "GPT-OSS-120B few-shot",
}
FIELDS = (
    "company",
    "role",
    "seniority",
    "skills",
    "min_experience_years",
    "max_experience_years",
    "salary",
    "locations",
    "work_mode",
    "employment_type",
    "education",
)
FIELD_SHORT = {
    "min_experience_years": "min exp",
    "max_experience_years": "max exp",
    "employment_type": "employment",
}


def _esc(s: Any) -> str:
    return html.escape(str(s))


def _pct(v: Any) -> str:
    return "-" if v is None else f"{100 * float(v):.1f}"


def _read(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def _measured(data: dict[str, Any], split: str) -> dict[str, dict[str, Any]]:
    return {r["approach"]: r for r in data.get("splits", {}).get(split, []) if r.get("status") == "measured"}


def _bar(v: float | None, label: str, *, cls: str = "") -> str:
    if v is None:
        return f'<td data-label="{label}"><span class="jx-bar__val sb-faint">-</span></td>'
    w = max(0.0, min(100.0, 100 * v))
    return (
        f'<td data-label="{label}" title="{label}: {100 * v:.1f}"><span class="jx-bar {cls}" style="--w:{w:.2f}">'
        f'<span class="jx-bar__fill"></span><span class="jx-bar__val">{100 * v:.1f}</span></span></td>'
    )


def _figure(title: str, sub: str, body: str, foot: str, *, wide: bool = True, fid: str = "") -> str:
    idattr = f' id="{fid}"' if fid else ""
    return f"""<figure class="sb-card sb-card--chart jx-chart{" jx-chart--wide" if wide else ""} sb-reveal"{idattr}>
  <figcaption><p class="sb-card__title">{title}</p><p class="sb-card__sub">{sub}</p></figcaption>
  {body}
  <div class="sb-card__foot">{foot}</div>
</figure>"""


# ------------------------------------------------------------------------------------------------------------
# Real postings vs synthetic


def real_vs_synth(data: dict[str, Any]) -> str:
    real, synth = _measured(data, "test_real"), _measured(data, "test_synth")
    if len(real) < 2:
        return ""
    rows = []
    order = sorted(real, key=lambda a: -float(real[a]["metrics"]["micro_f1"]))
    n_real = max(int(r.get("n") or 0) for r in real.values())
    for a in order:
        rr = real[a]["metrics"]
        sr = synth.get(a, {}).get("metrics")
        ours = "lora" in a
        if sr:
            d = 100 * (float(rr["micro_f1"]) - float(sr["micro_f1"]))
            drop = f"{d:+.1f}".replace("-", "&minus;")
        else:
            drop = "-"
        rows.append(
            f'<tr class="{"is-ours" if ours else ""}"><th scope="row"><span class="jx-bars__name">'
            f"{_esc(SHORT.get(a, a))}</span></th>"
            + _bar(float(sr["micro_f1"]) if sr else None, "Synthetic micro-F1 %", cls="jx-bar--muted")
            + _bar(float(rr["micro_f1"]), "Real micro-F1 %")
            + f'<td class="num jx-drop" data-label="Change (pp)">{drop}</td>'
            + _bar(
                None if rr.get("hallucination_rate") is None else float(rr["hallucination_rate"]),
                "Real hallucinated %",
                cls="jx-bar--warn",
            )
            + "</tr>"
        )
    body = f"""<table class="jx-bars jx-bars--4">
    <thead><tr><th scope="col">Approach</th><th scope="col">Synthetic micro-F1 %</th><th scope="col">Real micro-F1 %</th>
      <th scope="col" class="num">Real &minus; synthetic (pp)</th><th scope="col">Real hallucinated %</th></tr></thead>
    <tbody>{"".join(rows)}</tbody></table>"""
    sub = (
        f"The same models on <code>test_synth</code> (held-out synthetic families) and on <code>test_real</code>: {n_real} real "
        "Indian postings from public sources, labels AI-assisted and not human-verified. The gap is the honest measure of "
        "how far the synthetic training data transfers."
    )
    return _figure(
        "Synthetic vs real postings",
        sub,
        body,
        "Source: results/results.json (test_synth, test_real)",
        fid="real-vs-synth",
    )


def field_heat(data: dict[str, Any], split: str = "test_real") -> str:
    rows_by = _measured(data, split)
    shown = [a for a in FIELD_SET if a in rows_by]
    if len(shown) < 2:
        return ""
    head = "".join(f'<th scope="col" class="num">{_esc(FIELD_SHORT.get(f, f))}</th>' for f in FIELDS)
    body = []
    for a in shown:
        pf = rows_by[a].get("per_field") or {}
        cells = []
        for f in FIELDS:
            v = (pf.get(f) or {}).get("f1")
            if v is None:
                cells.append('<td class="num sb-faint">-</td>')
            else:
                h = max(0.0, min(1.0, float(v))) * 42
                cells.append(f'<td class="num" style="--h:{h:.0f}%">{100 * float(v):.0f}</td>')
        ours = "lora" in a
        body.append(
            f'<tr{" class=is-ours" if ours else ""}><th scope="row">{_esc(SHORT.get(a, a))}</th>{"".join(cells)}</tr>'
        )
    table = f"""<div class="sb-table-wrap"><table class="sb-table jx-table jx-heat jx-fieldheat">
    <thead><tr><th scope="col">Approach</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>"""
    kind = "real postings" if split == "test_real" else "synthetic held-out split"
    return _figure(
        f"Per-field F1 on the {kind}",
        "F1 % per field (list fields element-wise). Darker = higher. Read company, role and skills with the annotation "
        "caveats in mind: real titles and skill lists are often ambiguous, and accepted alternatives are scored as correct.",
        table,
        f"Source: results/results.json ({split}, per_field)",
        fid="field-f1",
    )


# ------------------------------------------------------------------------------------------------------------
# Robustness


ROB_LABEL = {
    "missing_salary": "Missing salary",
    "multiple_salaries": "Several salaries",
    "conflicting_salary": "Conflicting salary",
    "monthly_vs_annual": "Monthly vs annual",
    "multiple_locations": "Several locations",
    "work_mode": "Work-mode phrasing",
    "contradictory_experience": "Contradictory experience",
    "no_title": "No job title",
    "no_company": "No company",
    "mixed_language": "Hindi / English mix",
    "very_long": "Very long (~7k chars)",
    "malformed": "Malformed (HTML, caps, no punctuation)",
    "irrelevant": "Not a job post",
    "prompt_injection": "Prompt injection",
}


def robustness_table(root: Path) -> str:
    rep = _read(root / "results/robustness/summary.json")
    if not rep:
        return ""
    approaches = [a for a in FIELD_SET if a in rep] + [a for a in rep if a not in FIELD_SET]
    head = "".join(f'<th scope="col" class="num">{_esc(SHORT.get(a, a))}</th>' for a in approaches)
    rows = []
    for cat, label in ROB_LABEL.items():
        cells = []
        for a in approaches:
            m = rep[a].get(cat)
            if not m:
                cells.append('<td class="num sb-faint">-</td>')
                continue
            v = m.get("all_null_rate") if cat == "irrelevant" else m.get("micro_f1")
            if v is None:
                cells.append('<td class="num sb-faint">-</td>')
            else:
                cells.append(
                    f'<td class="num" style="--h:{max(0.0, min(1.0, float(v))) * 42:.0f}%">{100 * float(v):.0f}</td>'
                )
        rows.append(f'<tr><th scope="row">{label}</th>{"".join(cells)}</tr>')
    asr = []
    for a in approaches:
        m = rep[a].get("prompt_injection") or {}
        v = m.get("attack_success_rate")
        asr.append(
            '<td class="num sb-faint">-</td>'
            if v is None
            else f'<td class="num jx-asr" style="--h:{float(v) * 42:.0f}%">{100 * float(v):.0f}</td>'
        )
    rows.append(
        f'<tr class="jx-asr-row"><th scope="row">Injection success (lower is better)</th>{"".join(asr)}</tr>'
    )
    table = f"""<div class="sb-table-wrap"><table class="sb-table jx-table jx-heat jx-robust">
    <thead><tr><th scope="col">Category (12 cases each)</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>"""
    return _figure(
        "Robustness and prompt injection",
        "Micro-F1 % per category of the generated adversarial suite (gold by construction; companies from the held-out "
        "bucket). &ldquo;Not a job post&rdquo; shows the share of outputs with every field empty. The last row is the share "
        "of injection cases where an injected value (e.g. company &ldquo;HACKED&rdquo;) reached the output.",
        table,
        "Source: results/robustness/summary.json (jobx robustness-report)",
        fid="robustness",
    )


# ------------------------------------------------------------------------------------------------------------
# Calibration


def _reliability_svg(raw: dict[str, Any], cal: dict[str, Any]) -> str:
    size, pad = 220, 28
    inner = size - 2 * pad

    def xy(c: float, a: float) -> tuple[float, float]:
        return pad + c * inner, size - pad - a * inner

    def pts(rel: dict[str, Any], cls: str) -> str:
        bins = rel.get("bins") or []
        out = []
        line = []
        nmax = max((b["n"] for b in bins), default=1)
        for b in bins:
            x, y = xy(float(b["conf"]), float(b["acc"]))
            r = 2.5 + 4.5 * (b["n"] / nmax) ** 0.5
            line.append(f"{x:.1f},{y:.1f}")
            out.append(
                f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}"><title>confidence {b["conf"]:.2f}, accuracy {b["acc"]:.2f}, n={b["n"]}</title></circle>'
            )
        return f'<polyline class="{cls}-line" points="{" ".join(line)}"/>' + "".join(out)

    grid = "".join(
        f'<line class="jx-grid" x1="{pad}" y1="{xy(0, t)[1]:.1f}" x2="{size - pad}" y2="{xy(0, t)[1]:.1f}"/>'
        f'<text class="jx-tick" x="{pad - 4}" y="{xy(0, t)[1] + 3:.1f}" text-anchor="end">{t:g}</text>'
        f'<text class="jx-tick" x="{xy(t, 0)[0]:.1f}" y="{size - pad + 12}" text-anchor="middle">{t:g}</text>'
        for t in (0, 0.5, 1)
    )
    return f"""<svg class="jx-rel" viewBox="0 0 {size} {size}" role="img" aria-label="Reliability diagram: accuracy against confidence per bin">
  {grid}<line class="jx-diag" x1="{pad}" y1="{size - pad}" x2="{size - pad}" y2="{pad}"/>
  {pts(raw, "jx-raw")}{pts(cal, "jx-cal")}
  <text class="jx-axis" x="{size / 2}" y="{size - 2}" text-anchor="middle">confidence</text>
  <text class="jx-axis" x="9" y="{size / 2}" text-anchor="middle" transform="rotate(-90 9 {size / 2})">accuracy</text>
</svg>"""


def calibration_card(root: Path, approach: str = OURS) -> str:
    rep = _read(root / "results/calibration" / f"{approach}.json")
    if not rep:
        return ""
    panels = []
    for split in ("test_real", "test_synth", "robustness"):
        s = rep["splits"].get(split)
        if not s:
            continue
        raw, cal, at = s["raw_token_probability"], s["calibrated"], s["at_threshold"]
        stats = (
            f'<dl class="jx-calstats"><div><dt>ECE raw &rarr; calibrated</dt><dd>{raw["ece"]:.3f} &rarr; {cal["ece"]:.3f}</dd></div>'
            f"<div><dt>AUROC (correct vs wrong)</dt><dd>{cal['auroc'] if cal['auroc'] is not None else '-'}</dd></div>"
            f"<div><dt>Kept at threshold {rep['threshold']:.2f}</dt><dd>{_pct(at['coverage'])}% of fields</dd></div>"
            f"<div><dt>Accuracy kept / all</dt><dd>{_pct(at['selective_accuracy'])}% / {_pct(at['accuracy_all_fields'])}%</dd></div></dl>"
        )
        title = {
            "test_real": "Real postings",
            "test_synth": "Synthetic held-out",
            "robustness": "Robustness suite",
        }[split]
        panels.append(
            f'<div class="jx-calpanel"><p class="jx-calpanel__t">{title} <span class="sb-faint">'
            f"({s['n_examples']} posts, {s['n_fields']} fields)</span></p>{_reliability_svg(raw, cal)}{stats}</div>"
        )
    if not panels:
        return ""
    key = '<div class="jx-key" aria-hidden="true"><span><i class="jx-sw jx-sw--raw"></i>Raw token probability</span><span><i class="jx-sw jx-sw--ours"></i>Calibrated (fitted on val)</span><span><i class="jx-dash"></i>Perfect calibration</span></div>'
    sub = (
        f"Field-level confidence of the fine-tune (Q4_K_M): token log-probs of each value plus evidence support, mapped to "
        f"P(field correct) by a logistic model fitted on the validation split only ({rep['n_fields_val']} fields). "
        f"The abstention threshold ({rep['threshold']:.2f}) is the smallest one reaching {100 * rep['target_selective_accuracy']:.0f}% "
        "selective accuracy on validation; the held-out splits below were not used for fitting or for the threshold."
    )
    return _figure(
        "Is the confidence calibrated?",
        sub,
        key + f'<div class="jx-calgrid">{"".join(panels)}</div>',
        f"Source: results/calibration/{approach}.json (jobx calibrate)",
        fid="calibration",
    )


# ------------------------------------------------------------------------------------------------------------
# Quantisation sweep


def quant_card(root: Path) -> str:
    sw = _read(root / "results/quant/sweep.json")
    if not sw or not sw.get("variants"):
        return ""
    vs = sw["variants"]
    smax = max(float(v["size_gb"]) for v in vs)
    tmax = max(float(v.get("gen_tokens_per_s") or 0) for v in vs) or 1.0
    rows = []
    for v in vs:
        acc = v.get("accuracy", {})
        real = (acc.get("test_real") or {}).get("micro_f1")
        synth = (acc.get("test_synth") or {}).get("micro_f1")
        jv = (acc.get("test_real") or acc.get("test_synth") or {}).get("json_validity")
        demo = v["quant"] == "Q4_K_M"
        tag = ' <span class="sb-tag jx-demo-tag">browser</span>' if demo else ""
        tok = float(v.get("gen_tokens_per_s") or 0)
        rows.append(
            f'<tr class="{"is-best" if demo else ""}"><th scope="row">{_esc(v["quant"])}{tag}</th>'
            f'<td class="num"><span class="jx-mini"><span style="--w:{100 * float(v["size_gb"]) / smax:.1f}%"></span></span>{v["size_gb"]:.3f}</td>'
            f'<td class="num">{_pct(synth)}</td><td class="num">{_pct(real)}</td><td class="num">{_pct(jv)}</td>'
            f'<td class="num"><span class="jx-mini jx-mini--2"><span style="--w:{100 * tok / tmax:.1f}%"></span></span>{tok:.1f}</td>'
            f'<td class="num">{v.get("prompt_tokens_per_s") or "-"}</td>'
            f'<td class="num">{(v.get("warm_latency_s") or {}).get("p50") or "-"}</td>'
            f'<td class="num">{v.get("startup_s", "-")}</td><td class="num">{v.get("peak_ram_gb") or "-"}</td></tr>'
        )
    sub_n = sw.get("subsets", {})
    sub = (
        f"Every variant rebuilt from the released merged weights and run one after another on the same runner "
        f"({_esc(sw.get('hardware', ''))}, llama.cpp {_esc(sw.get('llama_cpp') or '')}), greedy decoding, compact prompt. "
        f"Accuracy on {sub_n.get('test_synth', 0)} evenly spaced synthetic and {sub_n.get('test_real', 0)} real postings. "
        "Start-up = server launch to healthy with the page cache dropped."
    )
    table = f"""<div class="sb-table-wrap"><table class="sb-table jx-bench jx-quant">
    <thead><tr><th scope="col">GGUF</th><th scope="col" class="num">Size (GB)</th><th scope="col" class="num">Synth F1 %</th>
      <th scope="col" class="num">Real F1 %</th><th scope="col" class="num">JSON valid %</th><th scope="col" class="num">Gen tok/s</th>
      <th scope="col" class="num">Prompt tok/s</th><th scope="col" class="num">p50 latency (s)</th>
      <th scope="col" class="num">Start-up (s)</th><th scope="col" class="num">Peak RAM (GB)</th></tr></thead>
    <tbody>{"".join(rows)}</tbody></table></div>"""
    threads = sw.get("thread_sweep") or []
    if threads:
        t_rows = " &middot; ".join(
            f"{t['threads']} thread{'s' if t['threads'] > 1 else ''}: {t.get('gen_tokens_per_s') or '-'} tok/s, "
            f"p50 {(t.get('warm_latency_s') or {}).get('p50') or '-'} s"
            for t in threads
        )
        table += f'<p class="jx-chart-note">Threads ({_esc(threads[0]["quant"])}): {t_rows}.</p>'
    return _figure(
        "Quantisation: size, speed and accuracy on one machine",
        sub,
        table,
        f"Source: results/quant/sweep.json ({_esc(sw.get('date', ''))}, @{_esc(sw.get('commit', ''))})",
        fid="quantisation",
    )


# ------------------------------------------------------------------------------------------------------------
# Data and leakage


def data_card(root: Path) -> str:
    stats = _read(root / "data/processed/stats.json")
    real = _read(root / "data/real/real_meta.json")
    audit = _read(root / "results/audit/leakage.json")
    if not stats:
        return ""
    final = stats["pipeline"]["final"]
    tiles = [
        (f"{final['train']:,}", "Synthetic train", "8 template families"),
        (f"{final['val']:,}", "Synthetic val", "2 other families"),
        (f"{final['test_synth']:,}", "Synthetic test", "2 held-out families + companies"),
    ]
    if real:
        srcs = ", ".join(f"{k.split('/')[-1]} ({v['licence']})" for k, v in real.get("sources", {}).items())
        tiles.append((str(real["n"]), "Real postings", f"{len(real.get('sources', {}))} public sources"))
    rob = root / "data/robustness/robustness.jsonl"
    if rob.exists():
        tiles.append((str(sum(1 for _ in rob.open())), "Robustness cases", "14 categories"))
    tiles_html = "".join(
        f'<div class="sb-stat"><dt class="sb-stat__label">{label}</dt><dd class="sb-stat__value">{v}</dd>'
        f'<dd class="sb-stat__ctx">{ctx}</dd></div>'
        for v, label, ctx in tiles
    )
    notes = []
    if audit:
        g = audit["gates"]
        nn = audit["near_duplicates"]["nearest_train_jaccard"]
        ng = audit.get("ngram_overlap_with_train", {})
        notes.append(
            f"<li><strong>Leakage audit {'passed' if audit['ok'] else 'FAILED'}.</strong> Cross-split exact duplicates: "
            f"{g['max_cross_split_exact_duplicates']['observed']}; cross-split pairs with word-3-gram Jaccard &ge; 0.6: "
            f"{g['max_cross_split_pairs_jaccard_ge_0.6']['observed']}; closest train item to any test_synth item: Jaccard "
            f"{nn.get('test_synth', {}).get('max', '-')}; mean 8-gram overlap with train: "
            f"{_pct((ng.get('test_synth') or {}).get('mean'))}% (test_synth), {_pct((ng.get('test_real') or {}).get('mean'))}% "
            "(test_real). Template families and companies are disjoint across splits.</li>"
        )
    if real:
        ann = real.get("annotation", {})
        notes.append(
            f"<li><strong>Real postings:</strong> {srcs}. {ann.get('method', '')} "
            f"({ann.get('human_verified', 0)} of {real['n']} verified by a person); "
            f"{ann.get('with_alternatives', 0)} rows list accepted alternatives for ambiguous fields.</li>"
        )
    body = (
        f'<dl class="sb-stats jx-datastats">{tiles_html}</dl><ul class="jx-datanotes">{"".join(notes)}</ul>'
    )
    return _figure(
        "Data and leakage checks",
        "What the models were trained and tested on. Synthetic labels are exact by construction; real-posting labels are not.",
        body,
        "Sources: data/processed/stats.json, data/real/real_meta.json, results/audit/leakage.json (jobx audit)",
        fid="data",
    )


# ------------------------------------------------------------------------------------------------------------
# Base vs fine-tuned comparison data (precomputed outputs from committed runs)


def comparison_data(root: Path, *, n: int = 6, max_chars: int = 1800) -> dict[str, Any] | None:
    from jobx.data.build import read_jsonl
    from jobx.eval.metrics import score_example
    from jobx.eval.run import SPLIT_FILES

    for split in ("test_real", "test_synth"):
        bp = root / "results/predictions" / f"{BASE}__{split}.jsonl"
        op = root / "results/predictions" / f"{OURS}__{split}.jsonl"
        gp = root / SPLIT_FILES[split]
        if bp.exists() and op.exists() and gp.exists():
            break
    else:
        return None
    gold = {g["id"]: g for g in read_jsonl(gp)}
    base = {p["id"]: p for p in read_jsonl(bp)}
    ours = {p["id"]: p for p in read_jsonl(op)}
    ids = [i for i in gold if i in base and i in ours and len(gold[i]["text"]) <= max_chars]
    ids.sort(key=lambda i: hashlib.sha1(i.encode()).hexdigest())  # fixed rule, not hand-picked
    out = []
    for i in ids[:n]:
        g = gold[i]
        alts = g.get("label_alternatives")
        sb = score_example(i, g["label"], base[i].get("output") or "", alts)
        so = score_example(i, g["label"], ours[i].get("output") or "", alts)
        src = g.get("source") or {}
        out.append(
            {
                "id": i,
                "text": g["text"],
                "gold": g["label"],
                "source": src.get("dataset") or src.get("corpus") or g.get("meta", {}).get("family"),
                "licence": src.get("licence") or src.get("corpus_licence"),
                "base": {
                    "json_ok": sb.json_ok,
                    "data": sb.pred,
                    "correct": {f: s.correct for f, s in sb.fields.items()},
                },
                "ours": {
                    "json_ok": so.json_ok,
                    "data": so.pred,
                    "correct": {f: s.correct for f, s in so.fields.items()},
                },
            }
        )
    meta_b = _read(root / "results/predictions" / f"{BASE}__{split}.meta.json") or {}
    meta_o = _read(root / "results/predictions" / f"{OURS}__{split}.meta.json") or {}
    return {
        "split": split,
        "rule": f"the first {n} postings of at most {max_chars} characters in SHA-1(id) order",
        "base": {
            "name": "Qwen2.5-0.5B-Instruct zero-shot (base model)",
            "date": meta_b.get("date"),
            "commit": meta_b.get("commit"),
        },
        "ours": {
            "name": "LoRA fine-tune, GGUF Q4_K_M (the browser model)",
            "date": meta_o.get("date"),
            "commit": meta_o.get("commit"),
        },
        "examples": out,
    }


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
