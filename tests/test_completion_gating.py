"""Failed or partial runs are never scored as results."""

import json
import shutil
from pathlib import Path

import pytest

from jobx.eval.results_table import build_tables, model_mix_label, short_model_names
from jobx.eval.run import NoCompletedPredictions, evaluate_all, evaluate_file


def make_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / "data/processed").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "results/predictions").mkdir(parents=True)
    shutil.copy("data/processed/test_synth.jsonl", root / "data/processed/test_synth.jsonl")
    shutil.copy("configs/approaches.yaml", root / "configs/approaches.yaml")
    return root


def write_run(root: Path, approach: str, rows: list[dict], meta: dict) -> Path:
    p = root / "results/predictions" / f"{approach}__test_synth.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    p.with_name(p.name.replace(".jsonl", ".meta.json")).write_text(json.dumps(meta))
    return p


def gold_rows(root: Path, n: int) -> list[dict]:
    lines = (root / "data/processed/test_synth.jsonl").read_text().splitlines()[:n]
    return [{"id": e["id"], "output": json.dumps(e["label"])} for e in map(json.loads, lines)]


def test_refuses_to_score_runs_with_no_completions(tmp_path):
    root = make_root(tmp_path)
    rows = [{"id": r["id"], "output": "", "error": "NotFoundError"} for r in gold_rows(root, 5)]
    p = write_run(root, "llama-3.3-70b-groq_few_shot", rows, {"n_examples": 5, "n_completed": 0})
    with pytest.raises(NoCompletedPredictions):
        evaluate_file(p, root, n_boot=0)
    stale = root / "results/metrics/llama-3.3-70b-groq_few_shot__test_synth.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    assert evaluate_all(root / "results/predictions", root / "results/metrics", root, n_boot=0) == []
    assert not stale.exists()


def test_partial_or_error_heavy_runs_are_incomplete_and_kept_out(tmp_path):
    root = make_root(tmp_path)
    good = gold_rows(root, 20)
    rows = good[:18] + [
        {"id": r["id"], "output": "", "error": "ModelOverloaded"} for r in good[18:]
    ]  # 10% errors
    meta = {
        "n_examples": 20,
        "model": "llama-3.3-70b-versatile",
        "hardware": "API",
        "model_chain": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "qwen/qwen3-32b"],
        "models_used": {"llama-3.3-70b-versatile": 10, "openai/gpt-oss-120b": 8},
        "usage": {"requests": 20, "prompt_tokens": 1000, "completion_tokens": 100},
        "cost_usd": 0.0,
    }
    write_run(root, "llama-3.3-70b-groq_few_shot", rows, meta)
    write_run(root, "regex_baseline", gold_rows(root, 10), {"n_examples": 10})
    evaluate_all(root / "results/predictions", root / "results/metrics", root, n_boot=0)
    m = json.loads((root / "results/metrics/llama-3.3-70b-groq_few_shot__test_synth.json").read_text())
    assert m["completion"] == {
        "n_completed": 18,
        "n_expected": 20,
        "n_errors": 2,
        "error_rate": 0.1,
        "status": "incomplete",
    }
    md, data = build_tables(root)
    synth = {r["approach"]: r for r in data["splits"]["test_synth"]}
    assert synth["llama-3.3-70b-groq_few_shot"]["status"] == "incomplete"
    assert synth["regex_baseline"]["status"] == "measured"
    assert "incomplete (18/20)" in md and "Incomplete runs (not part of the comparison above)" in md
    assert "Llama-3.3-70B via Groq (llama-3.3-70b-versatile / gpt-oss-120b fallback) (API, few-shot)" in md
    per_field = md.split("#### Per-field exact match")[1].split("####")[0]
    assert "Gemini" not in per_field

    # partial (fewer rows than expected) but no errors: also incomplete
    write_run(root, "llama-3.3-70b-groq_few_shot", good[:15], {**meta, "n_examples": 20})
    evaluate_all(root / "results/predictions", root / "results/metrics", root, n_boot=0)
    md, _ = build_tables(root)
    assert "incomplete (15/20)" in md

    # complete, <= 5% errors: a normal measured row
    write_run(
        root, "llama-3.3-70b-groq_few_shot", good, {**meta, "models_used": {"llama-3.3-70b-versatile": 20}}
    )
    evaluate_all(root / "results/predictions", root / "results/metrics", root, n_boot=0)
    md, data = build_tables(root)
    row = {r["approach"]: r for r in data["splits"]["test_synth"]}["llama-3.3-70b-groq_few_shot"]
    assert row["status"] == "measured" and row["name"] == "Llama-3.3-70B via Groq (API, few-shot)"


def test_model_mix_labels():
    assert short_model_names(["gemini-3.8-flash", "gemini-3.7-flash"]) == {
        "gemini-3.8-flash": "3.8",
        "gemini-3.7-flash": "3.7",
    }
    chain = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash"]
    assert model_mix_label({"model_chain": chain, "models_used": {"gemini-3.7-flash": 5}}) == "3.7 fallback"
    assert model_mix_label({"model_chain": chain, "models_used": {"gemini-3.8-flash": 3}}) == "3.8"
    assert model_mix_label({}) is None
    groq = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "qwen/qwen3-32b"]
    assert short_model_names(groq)["openai/gpt-oss-120b"] == "gpt-oss-120b"
    assert (
        model_mix_label(
            {"model_chain": groq, "models_used": {"qwen/qwen3-32b": 2, "llama-3.3-70b-versatile": 9}}
        )
        == "llama-3.3-70b-versatile / qwen3-32b fallback"
    )
