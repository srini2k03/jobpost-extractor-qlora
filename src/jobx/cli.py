"""`jobx` command line."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Job-posting extraction toolkit.")


@app.command("schema")
def schema_cmd(out: Path = Path("schema/jobposting.schema.json")) -> None:
    """Export the JSON Schema of the extraction target."""
    from jobx.schema import json_schema

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_schema(), indent=2) + "\n")
    typer.echo(f"wrote {out}")


@app.command("build-data")
def build_data(
    config: Path = Path("configs/data.yaml"),
    out_dir: Path = Path("data/processed"),
    paraphrased: Path = Path("data/paraphrased/paraphrased_train.jsonl"),
    n: Annotated[int | None, typer.Option(help="override n_examples")] = None,
    card: Path = Path("DATA_CARD.md"),
) -> None:
    """Generate, validate, split, dedup; write JSONL, stats.json and DATA_CARD.md."""
    from jobx.data.build import build
    from jobx.data.datacard import write_datacard

    stats = build(config, out_dir, paraphrased=paraphrased, n_examples=n)
    write_datacard(out_dir / "stats.json", card, real_meta=Path("data/real/real_meta.json"))
    typer.echo(json.dumps(stats["pipeline"], indent=2))


@app.command("datacard")
def datacard(stats: Path = Path("data/processed/stats.json"), out: Path = Path("DATA_CARD.md")) -> None:
    """Re-render DATA_CARD.md from stats.json."""
    from jobx.data.datacard import write_datacard

    write_datacard(stats, out, real_meta=Path("data/real/real_meta.json"))


@app.command("regex-baseline")
def regex_baseline(
    split: list[str] = typer.Option(["val", "test_synth", "test_real"]),  # noqa: B008
    out_dir: Path = Path("results/predictions"),
) -> None:
    """Run the rule-based baseline on the given splits (skips splits that don't exist yet)."""
    import time

    from jobx.baselines.regex_extractor import predict
    from jobx.data.build import read_jsonl, write_jsonl
    from jobx.eval.run import SPLIT_FILES, git_commit

    for s in split:
        path = Path(SPLIT_FILES[s])
        if not path.exists():
            typer.echo(f"skip {s}: {path} not found")
            continue
        rows = []
        for e in read_jsonl(path):
            t0 = time.perf_counter()
            out = predict(e["text"])
            rows.append({"id": e["id"], "output": out, "latency_s": round(time.perf_counter() - t0, 5)})
        dest = out_dir / f"regex_baseline__{s}.jsonl"
        write_jsonl(dest, rows)
        meta = {
            "approach": "regex_baseline",
            "model": "rules",
            "decoding": "-",
            "hardware": "CPU",
            "date": datetime.now(UTC).strftime("%Y-%m-%d"),
            "commit": git_commit(),
        }
        dest.with_name(dest.name.removesuffix(".jsonl") + ".meta.json").write_text(
            json.dumps(meta, indent=2) + "\n"
        )
        typer.echo(f"wrote {dest} ({len(rows)} rows)")


@app.command("eval")
def eval_cmd(
    pred_dir: Path = Path("results/predictions"),
    out_dir: Path = Path("results/metrics"),
    n_boot: int = 1000,
) -> None:
    """Score every prediction file and write results/metrics/*.json."""
    from jobx.eval.run import evaluate_all

    for p in evaluate_all(pred_dir, out_dir, n_boot=n_boot):
        m = json.loads(p.read_text())["metrics"]
        typer.echo(
            f"{p.name}: n={m['n']} micro_f1={m['micro_f1']:.3f} json={m['json_validity']:.3f} "
            f"halluc={m['hallucination_rate']:.3f}"
        )


@app.command("results-table")
def results_table() -> None:
    """Rebuild results/results_table.md, results/results.json and the README block."""
    from jobx.eval.results_table import write_all

    write_all(".")
    typer.echo("wrote results/results_table.md and results/results.json; README updated")


@app.command("taxonomy")
def taxonomy(
    pred_dir: Path = Path("results/predictions"), out: Path = Path("results/error_taxonomy.json")
) -> None:
    """Failure-taxonomy counts for every prediction file."""
    from jobx.eval.run import SPLIT_FILES, parse_pred_name, score_file
    from jobx.eval.taxonomy import taxonomy_counts

    res = {}
    for p in sorted(pred_dir.glob("*__*.jsonl")):
        _, s = parse_pred_name(p)
        if not Path(SPLIT_FILES.get(s, "-")).exists():
            continue
        scores, _ = score_file(p)
        res[p.name.removesuffix(".jsonl")] = taxonomy_counts(scores)
    out.write_text(json.dumps(res, indent=2) + "\n")
    typer.echo(json.dumps(res, indent=2))


@app.command("charts")
def charts(out: Path = Path("results/charts")) -> None:
    """Micro-F1 and size/speed/accuracy charts from results/results.json (needs matplotlib)."""
    from jobx.eval.charts import draw_charts

    try:
        written = draw_charts(".", out)
    except ImportError:  # matplotlib not installed: charts are optional
        typer.echo("matplotlib not installed; skipping charts")
        return
    for p in written:
        typer.echo(f"wrote {p}")


@app.command("model-card")
def model_card(out: Path = Path("MODEL_CARD.md")) -> None:
    """Render MODEL_CARD.md for the CPU-route release (model-v1) from measured results."""
    from jobx.cpu.card import write_model_card

    typer.echo(f"wrote {write_model_card('.', out)}")


@app.command("site")
def site(out: Path = Path("site"), model_dir: Path | None = None) -> None:
    """Build the GitHub Pages site: results, charts and (if model parts are given) the in-browser demo."""
    from jobx.eval.site import build_site

    typer.echo(f"wrote {build_site('.', out, model_dir=model_dir)}")


@app.command("curves")
def curves(log_history: Path, out: Path = Path("results/training/curves.png")) -> None:
    """Plot training curves from a Trainer log_history.json."""
    from jobx.eval.curves import plot_curves

    typer.echo(f"wrote {plot_curves(log_history, out)}")


@app.command("fetch-real")
def fetch_real(
    config: Path = Path("configs/real_data.yaml"),
    out: Path = Path("data/real/raw_postings.jsonl"),
    allow_unverified_licence: bool = False,
) -> None:
    """Download ~200 real Indian postings from a permissively licensed HF dataset (needs Hub access)."""
    from jobx.data.fetch_real import fetch

    typer.echo(json.dumps(fetch(config, out, allow_unverified_licence=allow_unverified_licence), indent=2))


@app.command("verify")
def verify_cmd(
    draft: Path = Path("data/real/test_real_draft.jsonl"),
    out: Path = Path("data/real/test_real.jsonl"),
    verifier: str = "human",
) -> None:
    """Check and correct draft labels for the real test set, one posting at a time."""
    from jobx.verify import verify

    verify(draft, out, verifier=verifier)


@app.command("pull")
def pull(dest: Path = Path(".")) -> None:
    """Download notebook outputs (predictions, metrics, logs) from the HF artifacts dataset repo."""
    from jobx.hub import pull_artifacts

    typer.echo(f"pulled into {pull_artifacts(dest)}")


if __name__ == "__main__":
    app()
