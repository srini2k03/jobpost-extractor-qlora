"""`jobx bench report`: raw CSVs + manifests -> summary.csv, charts and an SLA recommendation.

Everything is computed from measured rows only. A config's display label, group and settings come from its
manifest (falling back to configs/bench/matrix.yaml); quality comes from the repo's own metrics files for
the config's free-decoding quality pass (``results/metrics/<approach>__test_synth.json``).

Recommender: for each config, the sustainable rate is the highest measured requests/s among levels of the
chosen workload whose TTFT p95 and E2E p95 meet the target with < 1 % errors. Serving ``rps`` then needs
``ceil(rps / sustainable)`` containers (assumes replicas scale linearly behind a load balancer); the cheapest
config by USD/hour wins, ties broken by headroom.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jobx.bench import plan
from jobx.bench.metrics import SLA, read_rows, summarize_rows, write_summary

RESULTS = Path("results/bench")
ENRICH_KEYS = ("display", "group", "engine", "quantization", "max_num_seqs", "enable_prefix_caching")
MAX_ERROR_RATE = 0.01


def raw_files(raw_dir: str | Path) -> list[Path]:
    d = Path(raw_dir)
    return sorted([*d.rglob("*.csv"), *d.rglob("*.csv.gz")])


def load_manifests(manifest_dir: str | Path) -> dict[str, dict[str, Any]]:
    out = {}
    for p in sorted(Path(manifest_dir).glob("*.json")):
        m = json.loads(p.read_text())
        out[str(m.get("config", p.stem))] = m
    return out


def quality_metrics(metrics_dir: str | Path, approach: str) -> dict[str, Any] | None:
    p = Path(metrics_dir) / f"{approach}__test_synth.json"
    if not p.exists():
        return None
    m: dict[str, Any] = json.loads(p.read_text())["metrics"]
    return m


def enrich(
    summaries: list[dict[str, Any]],
    matrix: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    metrics_dir: str | Path,
) -> list[dict[str, Any]]:
    cfgs = {c.name: c for c in plan.expand(matrix)}
    for s in summaries:
        name = str(s.get("label", ""))
        man = manifests.get(name) or {}
        cfg = cfgs.get(name)
        settings = man.get("settings") or (cfg.to_dict() if cfg else {})
        s["config"] = name
        s["display"] = man.get("label") or (cfg.label if cfg and cfg.label else name)
        s["group"] = man.get("group") or settings.get("group", "")
        s["engine"] = man.get("engine") or settings.get("engine", "")
        s["quantization"] = settings.get("quantization") or ("gguf" if s["engine"] == "llama.cpp" else "none")
        s["max_num_seqs"] = settings.get("max_num_seqs")
        s["enable_prefix_caching"] = settings.get("enable_prefix_caching")
        s["engine_version"] = man.get("engine_version")
        s["gpu"] = man.get("gpu")
        approach = None
        if cfg is not None and cfg.quality_approach_ids():
            approach = cfg.quality_approach_ids()[0]
        q = quality_metrics(metrics_dir, approach) if approach else None
        s["quality_approach"] = approach
        s["micro_f1"] = q.get("micro_f1") if q else None
        s["json_validity"] = q.get("json_validity") if q else None
    return summaries


def build_summary(
    raw_dir: str | Path,
    manifest_dir: str | Path,
    metrics_dir: str | Path,
    matrix: dict[str, Any],
) -> list[dict[str, Any]]:
    sla_cfg = (matrix.get("load") or {}).get("sla") or {}
    sla = SLA(float(sla_cfg.get("ttft_s", 1.0)), float(sla_cfg.get("e2e_s", 5.0)))
    rate = plan.hourly_rate(matrix)["total"]
    rows = read_rows(raw_files(raw_dir))
    summaries = summarize_rows(rows, sla=sla, gpu_usd_per_hour=rate)
    order = {c.name: i for i, c in enumerate(plan.expand(matrix))}

    def key(s: dict[str, Any]) -> tuple[Any, ...]:
        conc = int(s["concurrency"]) if str(s.get("concurrency") or "").isdigit() else 0
        rate_v = float(s["rate"]) if s.get("rate") not in (None, "") else 0.0
        return (
            order.get(str(s["label"]), 999),
            str(s["label"]),
            str(s["workload"]),
            str(s["mode"]),
            conc,
            rate_v,
        )

    return sorted(enrich(summaries, matrix, load_manifests(manifest_dir), metrics_dir), key=key)


# ------------------------------------------------------------------------------------------- recommender


@dataclass
class Recommendation:
    config: str
    display: str
    workload: str
    sustainable_rps: float
    level: str
    ttft_p95_ms: float
    e2e_p95_ms: float
    replicas: int
    usd_per_hour: float
    usd_per_1k_requests: float
    micro_f1: float | None


def _f(v: Any) -> float | None:
    if v is None or v == "":
        return None
    return float(v)


def recommend(
    summaries: Sequence[dict[str, Any]],
    *,
    ttft_p95_s: float,
    e2e_p95_s: float,
    rps: float,
    workload: str = "extract-full",
    usd_per_hour: float | None = None,
) -> list[Recommendation]:
    """Configs able to serve ``rps`` within the latency targets, cheapest first (measured data only)."""
    best: dict[str, dict[str, Any]] = {}
    for s in summaries:
        if s.get("workload") != workload:
            continue
        ttft, e2e, thr = _f(s.get("ttft_p95_ms")), _f(s.get("e2e_p95_ms")), _f(s.get("requests_per_s"))
        err = _f(s.get("error_rate")) or 0.0
        if ttft is None or e2e is None or not thr or err >= MAX_ERROR_RATE:
            continue
        if ttft > ttft_p95_s * 1000 or e2e > e2e_p95_s * 1000:
            continue
        name = str(s.get("config") or s.get("label"))
        if name not in best or thr > float(best[name]["requests_per_s"]):
            best[name] = dict(s)
    out = []
    for name, s in best.items():
        thr = float(s["requests_per_s"])
        rate = usd_per_hour if usd_per_hour is not None else (_f(s.get("gpu_usd_per_hour")) or 0.0)
        replicas = max(1, math.ceil(rps / thr - 1e-9))
        level = f"c{s['concurrency']}" if s.get("mode") == "closed" else f"{s.get('rate')} req/s"
        out.append(
            Recommendation(
                config=name,
                display=str(s.get("display") or name),
                workload=workload,
                sustainable_rps=round(thr, 3),
                level=level,
                ttft_p95_ms=float(s["ttft_p95_ms"]),
                e2e_p95_ms=float(s["e2e_p95_ms"]),
                replicas=replicas,
                usd_per_hour=round(replicas * rate, 4),
                usd_per_1k_requests=round(replicas * rate / (rps * 3600) * 1000, 5) if rps > 0 else 0.0,
                micro_f1=_f(s.get("micro_f1")),
            )
        )
    return sorted(out, key=lambda r: (r.usd_per_hour, -r.sustainable_rps, r.config))


# ------------------------------------------------------------------------------------------- charts

PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be185d", "#4d7c0f", "#64748b"]


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "legend.frameon": False,
        }
    )
    return plt


def _closed(summaries: Sequence[dict[str, Any]], workload: str) -> dict[str, list[dict[str, Any]]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for s in summaries:
        if s.get("workload") == workload and s.get("mode") == "closed" and _f(s.get("output_tokens_per_s")):
            by.setdefault(str(s.get("display") or s["label"]), []).append(s)
    for rows in by.values():
        rows.sort(key=lambda r: int(r["concurrency"]))
    return by


def chart_latency_throughput(summaries: Sequence[dict[str, Any]], workload: str, out: Path) -> Path | None:
    by = _closed(summaries, workload)
    if not by:
        return None
    plt = _plt()
    import matplotlib.ticker

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, (name, rows) in enumerate(by.items()):
        x = [float(r["output_tokens_per_s"]) for r in rows]
        for ax, key in zip(axes, ("e2e_p95_ms", "ttft_p95_ms"), strict=True):
            y = [float(r[key]) / 1000 for r in rows]
            ax.plot(x, y, marker="o", ms=3, lw=1.4, color=PALETTE[i % len(PALETTE)], label=name)
    for ax, title in zip(axes, ("E2E p95 (s)", "TTFT p95 (s)"), strict=True):
        ax.set_xlabel("output tokens/s (closed loop, one point per concurrency level)")
        ax.set_ylabel(title)
        ax.set_yscale("log")
        fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(fmt)
        ax.set_title(f"{title.split(' (')[0]} vs throughput ({workload})")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, plt, out / f"latency_throughput_{workload}.png")


def _best_cost(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    ok = [
        r for r in rows if _f(r.get("usd_per_1k_requests")) and str(r.get("sla_ok")) in ("True", "true", "1")
    ]
    pool = ok or [r for r in rows if _f(r.get("usd_per_1k_requests"))]
    return min(pool, key=lambda r: float(r["usd_per_1k_requests"])) if pool else None


def chart_cost(summaries: Sequence[dict[str, Any]], workload: str, out: Path) -> Path | None:
    by = _closed(summaries, workload)
    best = {k: b for k, v in by.items() if (b := _best_cost(v)) is not None}
    if not best:
        return None
    plt = _plt()
    items = sorted(best.items(), key=lambda kv: float(kv[1]["usd_per_1k_requests"]), reverse=True)
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(items) + 1.2))
    vals = [float(b["usd_per_1k_requests"]) for _, b in items]
    ax.barh([k for k, _ in items], vals, color=PALETTE[0])
    for y, (v, (_, b)) in enumerate(zip(vals, items, strict=True)):
        ax.text(v, y, f" ${v:.4f} (c{b['concurrency']})", va="center", fontsize=7)
    ax.set_xlabel("USD per 1K requests (cheapest closed-loop level meeting the SLA; GPU+CPU+RAM)")
    ax.set_title(f"Cost per 1K requests ({workload})")
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    return _save(fig, plt, out / f"cost_per_1k_{workload}.png")


def prefix_pairs(summaries: Sequence[dict[str, Any]], matrix: dict[str, Any]) -> list[tuple[str, str]]:
    """(caching on, caching off) config pairs: same settings except enable_prefix_caching."""
    cfgs = {c.name: c for c in plan.expand(matrix)}
    present = {str(s.get("config") or s.get("label")) for s in summaries}
    keys = (
        "engine",
        "model",
        "revision",
        "quantization",
        "dtype",
        "max_num_seqs",
        "gpu_memory_utilization",
        "lora",
    )

    def serving(c: plan.ServeConfig) -> tuple[Any, ...]:
        d = c.to_dict()
        return tuple(json.dumps(d[k], sort_keys=True) for k in keys)

    pairs = []
    for off in cfgs.values():
        if off.enable_prefix_caching or off.name not in present:
            continue
        for on in cfgs.values():
            if on.enable_prefix_caching and on.name in present and serving(on) == serving(off):
                pairs.append((on.name, off.name))
    return pairs


def prefix_comparison(
    summaries: Sequence[dict[str, Any]], matrix: dict[str, Any], workload: str = "extract-full"
) -> list[dict[str, Any]]:
    """TTFT p95 with caching on vs off at every concurrency both configs measured."""
    idx = {
        (str(s.get("config") or s["label"]), int(s["concurrency"])): s
        for s in summaries
        if s.get("workload") == workload and s.get("mode") == "closed"
    }
    out = []
    for on, off in prefix_pairs(summaries, matrix):
        for (name, conc), s_on in sorted(idx.items(), key=lambda kv: kv[0][1]):
            s_off = idx.get((off, conc))
            if name != on or s_off is None:
                continue
            a, b = _f(s_on.get("ttft_p95_ms")), _f(s_off.get("ttft_p95_ms"))
            if a is None or b is None or b <= 0:
                continue
            out.append({"on": on, "off": off, "concurrency": conc, "ttft_p95_on_ms": a, "ttft_p95_off_ms": b,
                        "reduction": round(1 - a / b, 4)})  # fmt: skip
    return out


def chart_prefix(comp: Sequence[dict[str, Any]], out: Path) -> Path | None:
    if not comp:
        return None
    plt = _plt()
    pairs = list(dict.fromkeys((c["on"], c["off"]) for c in comp))
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.5 * len(pairs), 3.6), squeeze=False)
    for ax, (on, off) in zip(axes[0], pairs, strict=True):
        rows = [c for c in comp if c["on"] == on]
        x = list(range(len(rows)))
        ax.bar(
            [i - 0.2 for i in x],
            [r["ttft_p95_on_ms"] for r in rows],
            0.4,
            color=PALETTE[0],
            label="caching on",
        )
        ax.bar(
            [i + 0.2 for i in x],
            [r["ttft_p95_off_ms"] for r in rows],
            0.4,
            color=PALETTE[8],
            label="caching off",
        )
        ax.set_xticks(x, [f"c{r['concurrency']}" for r in rows])
        ax.set_ylabel("TTFT p95 (ms)")
        ax.set_title(f"Prefix caching: {on} vs {off}")
        ax.legend()
    fig.tight_layout()
    return _save(fig, plt, out / "prefix_caching_ttft.png")


def chart_quality(summaries: Sequence[dict[str, Any]], workload: str, out: Path) -> Path | None:
    pts: dict[str, tuple[float, float]] = {}
    for s in summaries:
        f1, thr = _f(s.get("micro_f1")), _f(s.get("output_tokens_per_s"))
        if s.get("workload") != workload or f1 is None or not thr:
            continue
        name = str(s.get("display") or s["label"])
        if name not in pts or thr > pts[name][0]:
            pts[name] = (thr, f1)
    if not pts:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, (name, (thr, f1)) in enumerate(pts.items()):
        ax.scatter([thr], [100 * f1], color=PALETTE[i % len(PALETTE)], s=30, zorder=3)
        ax.annotate(name, (thr, 100 * f1), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel(f"peak output tokens/s ({workload}, any level)")
    ax.set_ylabel("micro F1 % (test_synth, greedy)")
    ax.set_title("Quality vs throughput per quantization")
    fig.tight_layout()
    return _save(fig, plt, out / "quality_vs_throughput.png")


def _save(fig: Any, plt: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)
    return path


def draw_charts(summaries: Sequence[dict[str, Any]], matrix: dict[str, Any], out: Path) -> list[Path]:
    workloads = list(dict.fromkeys(str(s["workload"]) for s in summaries))
    paths: list[Path | None] = []
    for w in workloads:
        paths += [chart_latency_throughput(summaries, w, out), chart_cost(summaries, w, out)]
    paths.append(chart_prefix(prefix_comparison(summaries, matrix), out))
    paths.append(chart_quality(summaries, "extract-full", out))
    return [p for p in paths if p is not None]


# ------------------------------------------------------------------------------------------- entry point


def report(
    *,
    raw_dir: str | Path = RESULTS / "raw",
    manifest_dir: str | Path = RESULTS / "manifests",
    metrics_dir: str | Path = "results/metrics",
    out_dir: str | Path = RESULTS,
    matrix_path: str | Path = plan.DEFAULT_MATRIX,
    charts: bool = True,
) -> dict[str, Any]:
    """Write summary.csv, recommendation.json and charts/ under ``out_dir``; returns what was written."""
    matrix = plan.load_matrix(matrix_path)
    out = Path(out_dir)
    summaries = build_summary(raw_dir, manifest_dir, metrics_dir, matrix)
    if not summaries:
        return {"summary": None, "charts": [], "recommendation": None}
    summary = write_summary(out / "summary.csv", summaries)
    target = matrix.get("recommend") or {}
    recs = recommend(
        summaries,
        ttft_p95_s=float(target.get("ttft_p95_s", 1.0)),
        e2e_p95_s=float(target.get("e2e_p95_s", 5.0)),
        rps=float(target.get("rps", 10.0)),
        workload=str(target.get("workload", "extract-full")),
    )
    rec = {
        "target": {k: target.get(k) for k in ("ttft_p95_s", "e2e_p95_s", "rps", "workload")},
        "recommended": asdict(recs[0]) if recs else None,
        "candidates": [asdict(r) for r in recs],
        "prefix_caching": prefix_comparison(summaries, matrix),
    }
    rec_path = out / "recommendation.json"
    rec_path.write_text(json.dumps(rec, indent=2) + "\n")
    drawn: list[Path] = []
    if charts:
        try:
            drawn = draw_charts(summaries, matrix, out / "charts")
        except ImportError:  # matplotlib is optional
            drawn = []
    return {"summary": summary, "charts": drawn, "recommendation": rec_path}
