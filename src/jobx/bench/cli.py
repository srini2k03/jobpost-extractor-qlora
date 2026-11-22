"""`jobx bench ...`: serving load tests (see docs/serving_benchmark.md)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

bench_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Serving benchmark: load tests against OpenAI-compatible servers.",
)

DEFAULT_MATRIX = Path("configs/bench/matrix.yaml")


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


@bench_app.command("run")
def run(
    model: Annotated[
        str, typer.Option(help="model name the server expects (vLLM: --served-model-name or LoRA name)")
    ],
    out: Annotated[Path, typer.Option(help="raw per-request CSV, e.g. results/bench/raw/<name>.csv")],
    base_url: Annotated[str, typer.Option(help="OpenAI-compatible base URL")] = "http://127.0.0.1:8000/v1",
    workload: Annotated[
        str, typer.Option(help="extract-full | extract-compact | synthetic-long[-compact]")
    ] = "extract-full",
    concurrency: Annotated[str, typer.Option(help="closed-loop levels, comma-separated")] = "1,2,4,8",
    rate: Annotated[str, typer.Option(help="open-loop Poisson rates in req/s, comma-separated")] = "",
    num_requests: Annotated[int, typer.Option(help="measured requests per level")] = 64,
    duration: Annotated[float | None, typer.Option(help="seconds per level instead of a fixed count")] = None,
    warmup: Annotated[int, typer.Option(help="warm-up requests per level (excluded)")] = 4,
    max_concurrency: Annotated[int | None, typer.Option(help="open-loop cap on requests in flight")] = None,
    seed: int = 0,
    max_tokens: int = 384,
    output_median: Annotated[int, typer.Option(help="synthetic workloads: median output tokens")] = 256,
    output_cap: Annotated[int, typer.Option(help="synthetic workloads: max output tokens")] = 1024,
    label: Annotated[str | None, typer.Option(help="run label (default: output file stem)")] = None,
    api_key_env: Annotated[
        str | None, typer.Option(help="env var holding a bearer token, if the server needs one")
    ] = None,
    timeout: float = 600.0,
    trust_env: Annotated[
        bool, typer.Option(help="honour HTTP(S)_PROXY (off by default: proxies distort timing)")
    ] = False,
    gpu: Annotated[
        bool, typer.Option(help="sample local GPU memory/utilisation (no-op without a GPU)")
    ] = True,
) -> None:
    """Closed-loop sweep over concurrency levels and/or open-loop Poisson rates; writes one CSV row per request."""
    import asyncio

    from jobx.bench.gpu import GpuSampler
    from jobx.bench.loadgen import RunResult, StreamingClient, sweep, write_csv
    from jobx.bench.metrics import markdown_table, summarize_rows
    from jobx.bench.workload import OutputLengths, build_workload

    levels_c, levels_r = _ints(concurrency), _floats(rate)
    if not levels_c and not levels_r:
        raise typer.BadParameter("give --concurrency and/or --rate")
    per_level = num_requests
    n_levels = len(levels_c) + len(levels_r)
    requests = build_workload(
        workload,
        (warmup + per_level) * n_levels,
        seed=seed,
        max_tokens=max_tokens,
        lengths=OutputLengths(median=output_median, cap=output_cap),
    )
    api_key = os.environ.get(api_key_env) if api_key_env else None
    # Connection pool: one connection per request in flight. Uncapped open loop -> unlimited, so the client
    # never queues requests itself (that would be counted as server latency).
    pool = None if levels_r and not max_concurrency else max([*levels_c, max_concurrency or 0, 1])

    async def go() -> list[RunResult]:
        async with StreamingClient(
            base_url, model, api_key=api_key, timeout_s=timeout, max_connections=pool, trust_env=trust_env
        ) as client:
            return await sweep(
                client,
                requests,
                label=label or out.stem,
                workload=workload,
                concurrencies=levels_c,
                rates=levels_r,
                per_level=per_level,
                warmup=warmup,
                duration_s=duration,
                seed=seed,
                max_concurrency=max_concurrency,
                gpu_sampler_factory=GpuSampler if gpu else None,
            )

    runs = asyncio.run(go())
    write_csv(out, runs)
    rows = [row for r in runs for row in r.rows()]
    typer.echo(markdown_table(summarize_rows(rows)))
    typer.echo(f"wrote {out} ({len(rows)} requests, {n_levels} levels)")


@bench_app.command("summarize")
def summarize(
    csvs: Annotated[list[Path], typer.Argument(help="raw CSVs from `jobx bench run`")],
    out: Path = Path("results/bench/summary.csv"),
    gpu_price: Annotated[float | None, typer.Option(help="GPU USD per hour (overrides --gpu)")] = None,
    gpu: Annotated[
        str | None, typer.Option(help="GPU name to look up in the matrix price table, e.g. L4")
    ] = None,
    matrix: Path = DEFAULT_MATRIX,
    sla_ttft: Annotated[float, typer.Option(help="goodput SLA: max TTFT seconds")] = 1.0,
    sla_e2e: Annotated[float, typer.Option(help="goodput SLA: max end-to-end seconds")] = 5.0,
    markdown: Annotated[Path | None, typer.Option(help="also write the Markdown table here")] = None,
) -> None:
    """One summary row per run: latency percentiles, throughput, goodput, GPU memory and cost."""
    from jobx.bench.cost import gpu_price as lookup_price
    from jobx.bench.metrics import SLA, markdown_table, read_rows, summarize_rows, write_summary

    price = gpu_price if gpu_price is not None else (lookup_price(matrix, gpu) if gpu else None)
    summaries = summarize_rows(read_rows(csvs), sla=SLA(sla_ttft, sla_e2e), gpu_usd_per_hour=price)
    write_summary(out, summaries)
    table = markdown_table(summaries)
    if markdown is not None:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(table + "\n")
    typer.echo(table)
    typer.echo(f"wrote {out} ({len(summaries)} runs)")


@bench_app.command("workload")
def workload_cmd(
    workload: str = "extract-full",
    num_requests: int = 256,
    seed: int = 0,
    tokenizer: Annotated[
        str | None, typer.Option(help="HF tokenizer id for token counts (needs transformers)")
    ] = None,
) -> None:
    """Print the prompt-length distribution of a workload (chars; tokens with --tokenizer)."""
    from jobx.bench.workload import build_workload, hf_token_counter, prompt_stats

    reqs = build_workload(workload, num_requests, seed=seed)
    counter = hf_token_counter(tokenizer) if tokenizer else None
    typer.echo(json.dumps({"workload": workload, **prompt_stats(reqs, counter)}, indent=2))


@bench_app.command("fake-server")
def fake_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    ttft: float = 0.05,
    itl: float = 0.01,
    output_tokens: int = 32,
    prefix_cache_factor: float = 1.0,
    max_num_seqs: int | None = None,
    fail_every: int = 0,
) -> None:
    """Run the fake OpenAI-compatible streaming server (for smoke tests of the harness)."""
    import asyncio

    from jobx.bench.fake_server import FakeConfig, serve

    cfg = FakeConfig(ttft, itl, output_tokens, prefix_cache_factor, max_num_seqs, fail_every)
    asyncio.run(serve(cfg, host, port))


@bench_app.command("plan")
def plan_cmd(
    config: Annotated[str, typer.Option(help="'all', comma-separated names or group:<name>")] = "",
    request: Annotated[
        Path | None, typer.Option(help="request file, e.g. .github/bench-request.json")
    ] = None,
    levels: Annotated[str, typer.Option(help='JSON level overrides, e.g. {"duration_s": 30}')] = "",
    yes: Annotated[bool, typer.Option(help="accept an estimate above the cap")] = False,
    max_gpu_minutes: Annotated[
        float | None, typer.Option(help="cap (default: matrix max_gpu_minutes)")
    ] = None,
    matrix: Path = DEFAULT_MATRIX,
    github_output: Annotated[
        Path | None, typer.Option(help="append key=value outputs for GitHub Actions")
    ] = None,
    quality_only: Annotated[
        bool, typer.Option(help="only configs with a quality pass, no load test")
    ] = False,
) -> None:
    """Print the GPU-minute / dollar estimate of a selection; exit 2 if it exceeds the cap without --yes."""
    from jobx.bench import plan

    m = plan.load_matrix(matrix)
    req = json.loads(request.read_text()) if request else None
    if not config and req is None:
        config = "all"
    sel = plan.plan_selection(
        m,
        config=config,
        request=req,
        levels=json.loads(levels) if levels else None,
        yes=yes,
        cap_minutes=max_gpu_minutes,
        quality_only=quality_only,
    )
    typer.echo(plan.format_estimate(sel.decision, str(m.get("gpu", "L4"))))
    if github_output is not None:
        r = sel.request
        overrides = {**(r.overrides if r else {}), **(json.loads(levels) if levels else {})}
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"configs={','.join(c.name for c in sel.configs)}\n")
            f.write(f"levels={json.dumps(overrides, separators=(',', ':'))}\n")
            f.write(f"yes={'true' if (yes or (r and r.yes)) else 'false'}\n")
            f.write(f"skip_load={'true' if sel.skip_load else 'false'}\n")
            f.write(f"skip_quality={'true' if sel.skip_quality else 'false'}\n")
            f.write(f"minutes={sel.decision.total_minutes}\nusd={sel.decision.total_usd}\n")
            f.write(f"ok={'true' if sel.decision.ok else 'false'}\n")
    if not sel.decision.ok:
        raise typer.Exit(2)


@bench_app.command("report")
def report_cmd(
    raw_dir: Path = Path("results/bench/raw"),
    manifest_dir: Path = Path("results/bench/manifests"),
    metrics_dir: Path = Path("results/metrics"),
    out_dir: Path = Path("results/bench"),
    matrix: Path = DEFAULT_MATRIX,
    charts: Annotated[bool, typer.Option(help="draw charts (needs matplotlib)")] = True,
) -> None:
    """All raw CSVs + manifests -> results/bench/summary.csv, recommendation.json and charts/."""
    from jobx.bench.report import report

    res = report(
        raw_dir=raw_dir,
        manifest_dir=manifest_dir,
        metrics_dir=metrics_dir,
        out_dir=out_dir,
        matrix_path=matrix,
        charts=charts,
    )
    if res["summary"] is None:
        typer.echo(f"no raw CSVs under {raw_dir}: nothing to report (run bench.yml first)")
        return
    typer.echo(f"wrote {res['summary']}, {res['recommendation']}")
    for p in res["charts"]:
        typer.echo(f"wrote {p}")


@bench_app.command("recommend")
def recommend_cmd(
    ttft_p95: Annotated[float, typer.Option(help="target TTFT p95, seconds")] = 1.0,
    e2e_p95: Annotated[float, typer.Option(help="target end-to-end p95, seconds")] = 5.0,
    rps: Annotated[float, typer.Option(help="required requests per second")] = 10.0,
    workload: str = "extract-full",
    summary: Path = Path("results/bench/summary.csv"),
    top: int = 5,
) -> None:
    """Cheapest measured config that meets a TTFT p95 / E2E p95 target at a request rate."""
    from jobx.bench.metrics import read_rows
    from jobx.bench.report import recommend

    if not summary.exists():
        typer.echo(f"{summary} not found: run `jobx bench report` after a benchmark run")
        raise typer.Exit(1)
    recs = recommend(read_rows([summary]), ttft_p95_s=ttft_p95, e2e_p95_s=e2e_p95, rps=rps, workload=workload)
    if not recs:
        typer.echo("no measured config meets this target")
        raise typer.Exit(1)
    typer.echo(
        "| config | level | sustainable req/s | TTFT p95 ms | E2E p95 ms | replicas | USD/h | USD/1K req | F1 |"
    )
    typer.echo("|---|---|---|---|---|---|---|---|---|")
    for r in recs[:top]:
        f1 = "-" if r.micro_f1 is None else f"{100 * r.micro_f1:.1f}"
        typer.echo(
            f"| {r.display} | {r.level} | {r.sustainable_rps} | {r.ttft_p95_ms} | {r.e2e_p95_ms} | {r.replicas} | "
            f"{r.usd_per_hour:.3f} | {r.usd_per_1k_requests:.4f} | {f1} |"
        )
    typer.echo(f"recommended: {recs[0].display} ({recs[0].config})")
