"""Thin orchestration used by the Colab notebooks: run one approach on one split, score it,
save predictions + metrics in the repo layout and push them to the HF artifacts repo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jobx import REPO_ROOT
from jobx.baselines.hf_runner import run_predictions
from jobx.data.build import read_jsonl
from jobx.eval.run import SPLIT_FILES, eval_subset, evaluate_file, git_commit


def available_test_splits(root: Path = REPO_ROOT) -> list[str]:
    return [s for s in ("test_real", "test_synth") if (root / SPLIT_FILES[s]).exists()]


def push_results(paths: list[Path], root: Path = REPO_ROOT, private: bool = True) -> str | None:
    """Upload files to <user>/jobx-artifacts (dataset repo) under the same relative paths."""
    from huggingface_hub import HfApi

    from jobx.hub import repo_id, secret

    token = secret("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set: results kept locally only")
        return None
    api = HfApi(token=token)
    rid = repo_id("artifacts")
    api.create_repo(rid, repo_type="dataset", private=private, exist_ok=True)
    for p in paths:
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=str(Path(p).resolve().relative_to(root)),
            repo_id=rid,
            repo_type="dataset",
        )
    return rid


def is_done(metrics_path: Path, n_expected: int) -> bool:
    """A finished run: metrics exist, cover n_expected examples and the run is complete."""
    if not metrics_path.exists():
        return False
    try:
        d = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return False
    complete = (d.get("completion") or {}).get("status", "complete") == "complete"
    return bool(complete and d.get("metrics", {}).get("n") == n_expected)


def run_approach(
    model: Any,
    tok: Any,
    approach: str,
    split: str,
    *,
    model_id: str,
    shots: list[dict[str, Any]] | None = None,
    guided: bool = False,
    max_examples: int | None = None,
    batch_size: int = 8,
    max_new_tokens: int = 384,
    hardware: str | None = None,
    push: bool = True,
    skip_existing: bool = False,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    examples = eval_subset(read_jsonl(root / SPLIT_FILES[split]), max_examples)
    pred = root / "results" / "predictions" / f"{approach}__{split}.jsonl"
    mpath = root / "results" / "metrics" / f"{approach}__{split}.json"
    if skip_existing and is_done(mpath, len(examples)):
        print(f"{approach} / {split}: already done ({len(examples)} examples), skipping")
        done: dict[str, Any] = json.loads(mpath.read_text())
        return done
    if hardware is None:
        from jobx.baselines.hf_runner import gpu_name

        hardware = gpu_name()
    meta = {
        "approach": approach,
        "model": model_id,
        "hardware": hardware,
        "split": split,
        "commit": git_commit(root),
        "n_examples": len(examples),
    }
    run_predictions(
        model,
        tok,
        examples,
        pred,
        meta=meta,
        shots=shots or [],
        guided=guided,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    res = evaluate_file(pred, root)
    res.pop("evaluated_at", None)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    m = res["metrics"]
    print(
        f"{approach} / {split}: n={m['n']} micro F1={m['micro_f1']:.3f} JSON valid={m['json_validity']:.3f} "
        f"hallucination={(m['hallucination_rate'] or 0.0):.3f}"
    )
    if push:
        push_results([pred, Path(str(pred).removesuffix(".jsonl") + ".meta.json"), mpath], root)
    return res
