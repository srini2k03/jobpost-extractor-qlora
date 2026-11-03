"""Run the GPU steps on your own machine (same logic and output paths as the Colab notebooks).

    python -m jobx.local check                    # environment + GPU sanity check (PASS/FAIL table)
    python -m jobx.local paraphrase               # notebook 01
    python -m jobx.local baselines [--families qwen1.5b,qwen7b]   # notebook 02
    python -m jobx.local train [--sweep]          # notebook 03
    python -m jobx.local eval                     # notebook 04
    python -m jobx.local awq                      # notebook 05 (run inside .venv-awq; `make local-awq` does it)
    python -m jobx.local push                     # upload everything to the Hub, like the notebooks
    python -m jobx.local all                      # paraphrase -> baselines -> train -> eval (+ summary)

Steps are resumable: finished work is detected and skipped, so re-running after an interruption continues.
Nothing is pushed until `push` (or `--push` on a step).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from typing import Any

DEFAULT_PROFILE = "local_rtx4080"
# Estimated wall-clock on an RTX 4080 Laptop GPU (12 GB) under WSL2, first run (model downloads excluded).
EXPECTED_RUNTIME: dict[str, str] = {
    "check": "~1 min",
    "paraphrase": "~25-35 min (600 rewrites with the 7B 4-bit teacher + NLI filter + ~200 real drafts)",
    "baselines": "~60-80 min (1.5B: ~20 min, 7B 4-bit: ~45 min; guided decoding dominates)",
    "train": "~30-45 min (3 epochs, ~400 optimiser steps, validation JSON-F1 every 50 steps)",
    "eval": "~15-20 min (fine-tune predictions free + guided, adapter merge)",
    "awq": "~20-30 min (+ ~5 min to build .venv-awq the first time)",
    "push": "~5-10 min (uploads ~3 GB merged + ~1 GB AWQ + adapter + results)",
    "all": "~2.5-3.5 h for paraphrase -> baselines -> train -> eval, then run awq and push",
}
ALL_STEPS = ("paraphrase", "baselines", "train", "eval")


def _banner(step: str) -> None:
    print(f"\n=== {step} === expected on an RTX 4080 laptop: {EXPECTED_RUNTIME[step]}", flush=True)


def run_step(step: str, a: argparse.Namespace) -> Any:
    from jobx import pipeline

    prof = pipeline.load_profile(a.profile)
    if step == "paraphrase":
        return pipeline.paraphrase(prof, push=a.push, do_real_drafts=not a.no_real_drafts)
    if step == "baselines":
        fams = [f.strip() for f in a.families.split(",") if f.strip()]
        unknown = set(fams) - set(pipeline.BASELINE_FAMILIES)
        if unknown:
            raise SystemExit(
                f"unknown baseline families {sorted(unknown)}; choose from {list(pipeline.BASELINE_FAMILIES)}"
            )
        return pipeline.baselines(prof, families=fams, push=a.push)
    if step == "train":
        out = pipeline.train_adapter(a.config or prof.train_config, push=a.push, force=a.force)
        if a.sweep:
            pipeline.sweep(push=a.push)
        return str(out)
    if step == "eval":
        return str(pipeline.eval_finetuned(prof, adapter=a.adapter, push=a.push))
    if step == "awq":
        return str(pipeline.awq_benchmark(prof, push=a.push))
    if step == "push":
        return pipeline.push_outputs(prof, dry_run=a.dry_run)
    raise SystemExit(f"unknown step {step}")


def run_all(
    a: argparse.Namespace, runner: Callable[[str, argparse.Namespace], Any] = run_step
) -> list[dict[str, Any]]:
    summary = []
    for step in ALL_STEPS:
        _banner(step)
        t0 = time.time()
        try:
            runner(step, a)
            status = "ok"
        except Exception as e:  # keep going: later steps may still be useful, summary shows the failure
            status = f"FAILED: {type(e).__name__}: {str(e)[:200]}"
        summary.append({"step": step, "status": status, "minutes": round((time.time() - t0) / 60, 1)})
        if status != "ok" and step in ("train",):
            break  # eval needs the adapter
    print("\n=== summary ===")
    for s in summary:
        print(f"{s['step']:<11} {s['minutes']:>6} min  {s['status']}")
    print(
        "next: `make local-awq`, then `make local-push`; afterwards run the quantize-benchmark workflow (GGUF)"
    )
    return summary


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m jobx.local", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("step", choices=["check", *ALL_STEPS, "awq", "push", "all"])
    ap.add_argument("--profile", default=DEFAULT_PROFILE, help="name under configs/gpu/ or a YAML path")
    ap.add_argument(
        "--push", action="store_true", help="also upload this step's outputs (like the notebooks)"
    )
    ap.add_argument("--families", default="qwen1.5b,qwen7b", help="baselines: qwen1.5b,llama1b,qwen7b")
    ap.add_argument("--config", default=None, help="train: override the profile's training YAML")
    ap.add_argument("--sweep", action="store_true", help="train: also run the r/lr sweep (~15 min per run)")
    ap.add_argument("--force", action="store_true", help="train: retrain even if a finished adapter exists")
    ap.add_argument(
        "--adapter", default=None, help="eval: adapter dir or Hub repo (default: local run, else Hub)"
    )
    ap.add_argument(
        "--no-real-drafts", action="store_true", help="paraphrase: skip fetching/drafting real postings"
    )
    ap.add_argument("--requirements", default=None, help="check: requirements file to verify against")
    ap.add_argument("--no-bnb", action="store_true", help="check: skip the 4-bit load test")
    ap.add_argument("--dry-run", action="store_true", help="push: only list what would be uploaded")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    if a.step == "check":
        from jobx.local.check import main as check_main

        _banner("check")
        args = ["--no-bnb"] if a.no_bnb else []
        if a.requirements:
            args += ["--requirements", a.requirements]
        return check_main(args)
    if a.step == "all":
        summary = run_all(a)
        return 0 if all(s["status"] == "ok" for s in summary) else 1
    _banner(a.step)
    out = run_step(a.step, a)
    print(json.dumps(out, indent=2, default=str) if not isinstance(out, str) else out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
