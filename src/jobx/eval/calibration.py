"""Calibrate field-level confidence on the validation split; measure it on held-out splits.

    jobx calibrate --approach qwen2.5-0.5b-lora-cpu_q4_k_m [--target 0.95]

Inputs are prediction files written with ``--logprobs`` (``results/predictions/<approach>__<split>.jsonl``
rows carry ``field_logprobs``). For every (example, field) the outcome is whether the predicted field
is correct (exact match after normalisation, annotated alternatives accepted). The calibrator is fitted
on ``val`` only; the abstention threshold is the smallest one whose *validation* selective accuracy
reaches ``target``. ``test_synth`` / ``test_real`` / ``robustness`` are only measured, never tuned on.

Writes ``results/calibration/<approach>.json`` (reliability of raw token probabilities vs calibrated
confidence, risk-coverage, the chosen threshold) and ``results/calibration/calibrator.json`` (weights
used by the API for abstention).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from jobx.data.build import read_jsonl
from jobx.eval.metrics import score_example
from jobx.eval.run import SPLIT_FILES
from jobx.evidence import evidence_by_field, find_evidence
from jobx.schema.models import FIELD_ORDER, LIST_FIELDS
from jobx.uncertainty import Calibrator, choose_threshold, field_features, reliability, risk_coverage

EVAL_SPLITS: tuple[str, ...] = ("test_synth", "test_real", "robustness")
THRESHOLDS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def field_records(approach: str, split: str, root: Path) -> list[dict[str, Any]]:
    pred_path = root / "results/predictions" / f"{approach}__{split}.jsonl"
    gold_path = root / SPLIT_FILES[split]
    if not pred_path.exists() or not gold_path.exists():
        return []
    gold = {g["id"]: g for g in read_jsonl(gold_path)}
    out: list[dict[str, Any]] = []
    for p in read_jsonl(pred_path):
        g = gold.get(p["id"])
        lps = p.get("field_logprobs")
        if g is None or lps is None:
            continue
        s = score_example(p["id"], g["label"], p.get("output") or "", g.get("label_alternatives"))
        if not s.json_ok:
            continue
        ev = evidence_by_field(find_evidence(g["text"], s.pred))
        for f in FIELD_ORDER:
            v = s.pred.get(f)
            is_null = v is None or v == []
            out.append(
                {
                    "id": p["id"],
                    "field": f,
                    "correct": float(s.fields[f].correct),
                    "lp": lps.get(f),
                    "evidence": None if is_null else ev.get(f, "unsupported"),
                    "is_null": is_null,
                }
            )
    return out


def _xy(recs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    X = np.array([field_features(r["lp"], r["evidence"], r["is_null"]) for r in recs], dtype=float)
    y = np.array([r["correct"] for r in recs], dtype=float)
    raw = np.array([math.exp(r["lp"]["mean"]) if r["lp"] else 1.0 for r in recs], dtype=float)
    return X, y, [r["field"] for r in recs], raw


def calibrate(
    approach: str, root: str | Path = ".", *, target: float = 0.95, write: bool = True
) -> dict[str, Any]:
    root = Path(root)
    val = field_records(approach, "val", root)
    if not val:
        raise FileNotFoundError(f"no val predictions with field_logprobs for {approach}")
    Xv, yv, fv, _ = _xy(val)
    cal = Calibrator().fit(Xv, yv, fv)
    pv = cal.predict(Xv, fv)
    thr = choose_threshold(pv, yv, target)
    report: dict[str, Any] = {
        "approach": approach,
        "fitted_on": "val",
        "n_fields_val": len(val),
        "target_selective_accuracy": target,
        "threshold": round(thr, 4),
        "calibrator": cal.to_dict(),
        "splits": {},
    }
    for split in ("val", *EVAL_SPLITS):
        recs = val if split == "val" else field_records(approach, split, root)
        if not recs:
            continue
        X, y, fs, raw = _xy(recs)
        p = cal.predict(X, fs)
        keep = p >= thr
        report["splits"][split] = {
            "n_fields": len(recs),
            "n_examples": len({r["id"] for r in recs}),
            "raw_token_probability": reliability(raw, y),
            "calibrated": reliability(p, y),
            "risk_coverage": risk_coverage(p, y, (*THRESHOLDS, thr)),
            "at_threshold": {
                "coverage": round(float(keep.mean()), 4),
                "selective_accuracy": round(float(y[keep].mean()), 4) if keep.any() else None,
                "accuracy_all_fields": round(float(y.mean()), 4),
                "abstained_fields_accuracy": round(float(y[~keep].mean()), 4) if (~keep).any() else None,
            },
            "per_field_accuracy": {
                f: round(float(y[[i for i, x in enumerate(fs) if x == f]].mean()), 4)
                for f in FIELD_ORDER
                if f in fs
            },
            "list_fields": list(LIST_FIELDS),
        }
    if write:
        d = root / "results/calibration"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{approach}.json").write_text(json.dumps(report, indent=2) + "\n")
        (d / "calibrator.json").write_text(
            json.dumps({"approach": approach, "threshold": report["threshold"], **cal.to_dict()}, indent=2)
            + "\n"
        )
    return report
