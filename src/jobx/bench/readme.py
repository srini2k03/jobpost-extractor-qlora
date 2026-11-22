"""The README's "Serving benchmark (vLLM)" block, generated from committed files like the results tables.

Inputs: configs/bench/matrix.yaml (method, matrix, estimate), results/bench/summary.csv and
recommendation.json (written by `jobx bench report`), results/bench/manifests/ and results/metrics/.
Without a summary the block says the benchmark is pending; findings are only ever computed from
measured rows, never estimated.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import median
from typing import Any

from jobx.bench import plan

START, END = "<!-- BENCH:START -->", "<!-- BENCH:END -->"
QUANT_LABELS = {
    "fp8": "FP8 (dynamic)",
    "awq_marlin": "AWQ int4",
    "awq": "AWQ int4",
    "gptq_marlin": "GPTQ int4",
}
DTYPE_LABELS = {"bfloat16": "BF16", "float16": "FP16"}
PENDING_LINE = (
    "**Status: pending.** No serving numbers yet: run `bench.yml` after adding the Modal token "
    "(repository secrets `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`), by pushing `.github/bench-request.json`."
)


def _f(v: Any) -> float | None:
    if v is None or v == "":
        return None
    return float(v)


def _load_summary(root: Path) -> list[dict[str, str]]:
    p = root / "results/bench/summary.csv"
    if not p.exists():
        return []
    from jobx.bench.metrics import read_rows

    return read_rows([p])


def _levels(c: plan.ServeConfig) -> str:
    parts = []
    if c.concurrency:
        parts.append("c " + "/".join(str(x) for x in c.concurrency))
    if c.rates:
        parts.append("Poisson " + "/".join(f"{x:g}" for x in c.rates) + " req/s")
    return ", ".join(parts)


def matrix_table(matrix: dict[str, Any]) -> list[str]:
    cfgs = plan.expand(matrix)
    ests = {e["config"]: e for e in plan.budget(cfgs, matrix).estimates}
    md = [
        "| Config | Model | Engine / quantization | max_num_seqs | Prefix caching | Workloads | Levels | Quality | Est. L4 min |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cfgs:
        quant = QUANT_LABELS.get(c.quantization or "", c.quantization or "")
        if not quant:
            quant = "GGUF Q4_K_M" if c.engine == "llama.cpp" else DTYPE_LABELS.get(c.dtype, c.dtype)
        engine = f"{'vLLM' if c.engine == 'vllm' else c.engine} {quant}" + (" + LoRA" if c.lora else "")
        seqs = f"{c.parallel} slots" if c.engine == "llama.cpp" else str(c.max_num_seqs)
        apc = "-" if c.engine == "llama.cpp" else ("on" if c.enable_prefix_caching else "**off**")
        qual = ("free + guided" if c.quality_guided else "free") if c.quality else "-"
        md.append(
            f"| `{c.name}` | `{c.model}` | {engine} | {seqs} | {apc} | {', '.join(c.workloads)} | {_levels(c)} | "
            f"{qual} | {ests[c.name]['minutes']} |"
        )
    return md


def best_rows(summary: list[dict[str, str]], workload: str = "extract-full") -> dict[str, dict[str, str]]:
    """Per config: the level with the highest requests/s that met the SLA (else the highest overall)."""
    out: dict[str, dict[str, str]] = {}
    for config in dict.fromkeys(r["config"] for r in summary if r.get("workload") == workload):
        rows = [r for r in summary if r["config"] == config and r.get("workload") == workload]
        ok = [r for r in rows if r.get("sla_ok") in ("True", "true", "1")]
        pool = ok or rows
        out[config] = max(pool, key=lambda r: _f(r.get("requests_per_s")) or 0.0)
        out[config]["_sla"] = "yes" if ok else "no"
    return out


def peak_tokens(summary: list[dict[str, str]], config: str, workload: str = "extract-full") -> float | None:
    vals = [
        v
        for r in summary
        if r["config"] == config and r.get("workload") == workload and r.get("mode") == "closed"
        if (v := _f(r.get("output_tokens_per_s"))) is not None
    ]
    return max(vals) if vals else None


def findings(summary: list[dict[str, str]], rec: dict[str, Any] | None) -> list[str]:
    """Plain-language findings, each only when the rows it needs were measured."""
    out = []
    f1 = {r["config"]: _f(r.get("micro_f1")) for r in summary}
    base = peak_tokens(summary, "q15-bf16")
    names = {"q15-fp8": "FP8 (dynamic)", "q15-awq": "AWQ 4-bit", "q15-gptq": "GPTQ 4-bit"}
    for cfg, name in names.items():
        peak = peak_tokens(summary, cfg)
        if base and peak:
            s = f"- {name} gave **{peak / base:.2f}x** the peak output tokens/s of BF16 on `extract-full`"
            a, b = f1.get("q15-bf16"), f1.get(cfg)
            if a is not None and b is not None:
                drop = 100 * (a - b)
                if drop >= 0.05:
                    s += f", at **{drop:.1f}** points of micro-F1 drop"
                elif drop <= -0.05:
                    s += f", with micro F1 **{-drop:.1f}** points higher"
                else:
                    s += ", with no micro-F1 change"
            out.append(s + ".")
    comp = (rec or {}).get("prefix_caching") or []
    for on in dict.fromkeys(c["on"] for c in comp):
        reds = [c["reduction"] for c in comp if c["on"] == on]
        out.append(
            f"- Prefix caching cut TTFT p95 by **{100 * median(reds):.0f}%** (median over {len(reds)} concurrency "
            f"levels) for `{on}` on the shared system + 3-shot prompt."
        )
    q3, q3b = peak_tokens(summary, "q3-awq"), peak_tokens(summary, "q3-bf16")
    if q3 and q3b:
        out.append(f"- At 3B, AWQ gave **{q3 / q3b:.2f}x** the peak output tokens/s of BF16.")
    lcpp, awq = peak_tokens(summary, "q15-llamacpp-q4km"), peak_tokens(summary, "q15-awq")
    if lcpp and awq:
        out.append(
            f"- vLLM AWQ reached **{awq / lcpp:.2f}x** the peak output tokens/s of llama.cpp Q4_K_M on the same L4."
        )
    best = (rec or {}).get("recommended")
    if best:
        t = (rec or {})["target"]
        out.append(
            f"- Cheapest config for TTFT p95 <= {t['ttft_p95_s']} s, E2E p95 <= {t['e2e_p95_s']} s at {t['rps']} req/s "
            f"(`{t['workload']}`): **{best['display']}** ({best['replicas']} x L4, "
            f"${best['usd_per_1k_requests']:.4f} per 1K requests)."
        )
    return out


def results_table(summary: list[dict[str, str]], matrix: dict[str, Any], root: Path) -> list[str]:
    best = best_rows(summary)
    md = [
        "| Config | Best level (SLA met) | req/s | Output tok/s | TTFT p95 ms | E2E p95 ms | USD / 1K req | "
        "GPU mem peak (MiB) | Micro F1 % | Status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    manifests = {
        p.stem: json.loads(p.read_text()) for p in sorted((root / "results/bench/manifests").glob("*.json"))
    }
    for c in plan.expand(matrix):
        r = best.get(c.name)
        if r is None:
            md.append(f"| {c.label or c.name} | - | - | - | - | - | - | - | - | pending |")
            continue
        level = f"c{r['concurrency']}" if r.get("mode") == "closed" else f"{r.get('rate')} req/s"
        f1 = _f(r.get("micro_f1"))
        man = manifests.get(c.name) or {}
        date = str(man.get("started_utc", ""))[:10]
        md.append(
            f"| {c.label or c.name} | {level} ({r['_sla']}) | {r.get('requests_per_s')} | {r.get('output_tokens_per_s')} | "
            f"{r.get('ttft_p95_ms')} | {r.get('e2e_p95_ms')} | {r.get('usd_per_1k_requests') or '-'} | "
            f"{r.get('gpu_mem_peak_mib') or '-'} | {'-' if f1 is None else f'{100 * f1:.1f}'} | "
            f"measured {date} |"
        )
    return md


def bench_block(root: str | Path = ".") -> str:
    root = Path(root)
    matrix = plan.load_matrix(root / "configs/bench/matrix.yaml")
    cfgs = plan.expand(matrix)
    d = plan.budget(cfgs, matrix)
    rate = plan.hourly_rate(matrix)
    load = matrix.get("load") or {}
    summary = _load_summary(root)
    rec_path = root / "results/bench/recommendation.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() and summary else None
    md = [
        "_Generated by `jobx results-table` from `results/bench/` and `configs/bench/matrix.yaml`; details in "
        "[`docs/serving_benchmark.md`](docs/serving_benchmark.md)._",
        "",
    ]
    if not summary:
        md += [PENDING_LINE, ""]
    md += [
        f"**Method.** vLLM {matrix['vllm_version']} (OpenAI-compatible server) on one Modal {matrix.get('gpu', 'L4')} "
        "per config, with the streaming load generator (`jobx bench`) running in the same container. Workloads: "
        "the real extraction request (`SYSTEM_PROMPT` + 3 shots + posting, ~91% shared prefix) and a decode-heavy "
        "variant with seeded lognormal output lengths. Levels: closed loop at "
        f"{'/'.join(str(x) for x in load.get('concurrency', []))} concurrent requests and open-loop Poisson arrivals "
        f"at {'/'.join(f'{x:g}' for x in load.get('rates', []))} req/s, {load.get('duration_s', 60):g} s each after "
        "warm-up. Measured: TTFT, inter-token latency, E2E p50/p95/p99, throughput, goodput (TTFT <= "
        f"{load.get('sla', {}).get('ttft_s', 1)} s and E2E <= {load.get('sla', {}).get('e2e_s', 5)} s), GPU memory, "
        "KV-cache use, prefix-cache hit rate and USD per 1K requests. Quality: the repo's own eval on all 471 "
        "`test_synth` postings per quantization (rows in the results table above).",
        "",
        "**Matrix** ([`configs/bench/matrix.yaml`](configs/bench/matrix.yaml)):",
        "",
        *matrix_table(matrix),
        "",
        f"**Cost estimate.** {d.total_minutes} L4-minutes ({d.total_minutes / 60:.1f} h): **${d.total_usd_gpu:.2f}** "
        f"of GPU at ${rate['gpu']:.2f}/h, ${d.total_usd:.2f} including the container's CPU and memory "
        f"(${rate['total']:.2f}/h; Modal list prices, to verify). Refused above {d.cap_minutes:g} GPU-minutes unless "
        "`yes` is set.",
        "",
        "**Reproduce.** Add the `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` repository secrets, then push a change to "
        "[`.github/bench-request.json`](.github/bench-request.json) (or dispatch `bench`). Locally: "
        '`pip install -e ".[bench,modal]"`, `jobx bench plan`, `modal run -m jobx.bench.modal_app --matrix all`, '
        "`jobx bench report`.",
        "",
    ]
    if summary:
        md += ["**Results** (`extract-full`, per config the highest-throughput level that met the SLA):", ""]
        md += results_table(summary, matrix, root)
        md.append("")
        found = findings(summary, rec)
        if found:
            md += ["**Findings** (from the measured rows above):", "", *found, ""]
        charts = [
            ("latency vs throughput", "latency_throughput_extract-full.png"),
            ("cost per 1K requests", "cost_per_1k_extract-full.png"),
            ("prefix caching TTFT", "prefix_caching_ttft.png"),
            ("quality vs throughput", "quality_vs_throughput.png"),
        ]
        for alt, name in charts:
            if (root / "results/bench/charts" / name).exists():
                md += [f"![{alt}](results/bench/charts/{name})", ""]
    return "\n".join(md).rstrip() + "\n"


def update_readme(readme: str | Path, block: str) -> bool:
    p = Path(readme)
    text = p.read_text()
    if START not in text or END not in text:
        return False
    new = re.sub(
        re.escape(START) + r".*?" + re.escape(END), lambda _: f"{START}\n{block}{END}", text, flags=re.DOTALL
    )
    if new != text:
        p.write_text(new)
    return True
