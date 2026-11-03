"""Scoring of model outputs against gold labels.

Definitions (also in the README):
* json_validity   - output contains a parseable JSON object.
* schema_validity - that object validates against ``JobPosting`` exactly.
* field EM        - normalised prediction equals normalised gold (null == null counts as a match).
* scalar P/R/F1   - a stated (non-null) prediction is a TP if correct, else an FP; a stated gold value
                    that was missed or wrong is an FN.
* list P/R/F1     - element-wise over canonicalised skills / locations.
* micro F1        - TP/FP/FN pooled over all fields; macro F1 - mean of per-field F1.
* hallucination   - a field whose gold is null/[] but the prediction is non-null/non-empty,
                    divided by the number of gold-null fields.
Invalid or schema-invalid outputs are scored field by field from whatever could be parsed
(nothing, if JSON is invalid), so the field metrics include their cost.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from jobx.normalize.fields import normalise_field
from jobx.schema.models import FIELD_ORDER, LIST_FIELDS
from jobx.schema.parse import parse_output


@dataclass
class FieldScore:
    correct: bool
    tp: int
    fp: int
    fn: int
    gold_null: bool
    pred_null: bool

    @property
    def hallucinated(self) -> bool:
        return self.gold_null and not self.pred_null


@dataclass
class ExampleScore:
    id: str
    json_ok: bool
    schema_ok: bool
    truncated: bool
    fields: dict[str, FieldScore] = field(default_factory=dict)
    pred: dict[str, Any] = field(default_factory=dict)
    gold: dict[str, Any] = field(default_factory=dict)
    gold_norm: dict[str, Any] = field(default_factory=dict)
    pred_norm: dict[str, Any] = field(default_factory=dict)


def _is_null(v: Any) -> bool:
    return v is None or v == () or v == []


def score_field(name: str, gold: Any, pred: Any) -> FieldScore:
    g = normalise_field(name, gold)
    p = normalise_field(name, pred)
    if name in LIST_FIELDS:
        gs, ps = set(g), set(p)
        tp = len(gs & ps)
        return FieldScore(gs == ps, tp, len(ps - gs), len(gs - ps), not gs, not ps)
    gn, pn = _is_null(g), _is_null(p)
    correct = g == p
    tp = int(not pn and correct)
    fp = int(not pn and not correct)
    fn = int(not gn and not correct)
    return FieldScore(correct, tp, fp, fn, gn, pn)


def score_example(ex_id: str, gold: dict[str, Any], output: str) -> ExampleScore:
    pr = parse_output(output)
    s = ExampleScore(ex_id, pr.json_ok, pr.schema_ok, pr.truncated, pred=pr.data, gold=gold)
    for f in FIELD_ORDER:
        s.fields[f] = score_field(f, gold.get(f), pr.data.get(f))
        s.gold_norm[f] = normalise_field(f, gold.get(f))
        s.pred_norm[f] = normalise_field(f, pr.data.get(f))
    return s


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    if tp + fp + fn == 0:
        return float("nan"), float("nan"), float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn)
    return p, r, f1


@dataclass
class Arrays:
    """Per-example count arrays; aggregation is a function of column sums, so bootstrap is cheap."""

    json_ok: np.ndarray
    schema_ok: np.ndarray
    correct: np.ndarray  # (n, F)
    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    halluc: np.ndarray
    gold_null: np.ndarray
    all_correct: np.ndarray

    @classmethod
    def from_scores(cls, scores: Sequence[ExampleScore]) -> Arrays:
        def mat(attr: str) -> np.ndarray:
            return np.array([[getattr(s.fields[f], attr) for f in FIELD_ORDER] for s in scores], dtype=float)

        correct = mat("correct")
        return cls(
            json_ok=np.array([s.json_ok for s in scores], dtype=float),
            schema_ok=np.array([s.schema_ok for s in scores], dtype=float),
            correct=correct,
            tp=mat("tp"),
            fp=mat("fp"),
            fn=mat("fn"),
            halluc=np.array([[s.fields[f].hallucinated for f in FIELD_ORDER] for s in scores], dtype=float),
            gold_null=mat("gold_null"),
            all_correct=correct.min(axis=1) if len(scores) else np.zeros(0),
        )


def _summary(a: Arrays, idx: np.ndarray | None = None) -> dict[str, float]:
    def S(x: np.ndarray) -> np.ndarray:
        return x[idx].sum(axis=0) if idx is not None else x.sum(axis=0)

    n = len(a.json_ok) if idx is None else len(idx)
    tp, fp, fn = S(a.tp), S(a.fp), S(a.fn)
    per_f1 = np.array([_prf(tp[i], fp[i], fn[i])[2] for i in range(len(FIELD_ORDER))])
    micro = _prf(tp.sum(), fp.sum(), fn.sum())
    gnull = S(a.gold_null).sum()
    return {
        "json_validity": float(S(a.json_ok) / n),
        "schema_validity": float(S(a.schema_ok) / n),
        "field_em_mean": float((S(a.correct) / n).mean()),
        "record_exact_match": float(S(a.all_correct) / n),
        "micro_precision": micro[0],
        "micro_recall": micro[1],
        "micro_f1": micro[2],
        "macro_f1": float(np.nanmean(per_f1)),
        "hallucination_rate": float(S(a.halluc).sum() / gnull) if gnull else float("nan"),
    }


HEADLINE = (
    "micro_f1",
    "macro_f1",
    "field_em_mean",
    "json_validity",
    "schema_validity",
    "hallucination_rate",
    "record_exact_match",
)


def aggregate(scores: Sequence[ExampleScore], *, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    if not scores:
        raise ValueError("no scores")
    a = Arrays.from_scores(scores)
    n = len(scores)
    out: dict[str, Any] = {"n": n, **_summary(a)}
    per_field: dict[str, dict[str, float]] = {}
    tp, fp, fn = a.tp.sum(0), a.fp.sum(0), a.fn.sum(0)
    for i, f in enumerate(FIELD_ORDER):
        p, r, f1 = _prf(tp[i], fp[i], fn[i])
        gn = a.gold_null[:, i].sum()
        per_field[f] = {
            "em": float(a.correct[:, i].mean()),
            "precision": p,
            "recall": r,
            "f1": f1,
            "hallucination_rate": float(a.halluc[:, i].sum() / gn) if gn else float("nan"),
            "support_gold_stated": int(n - gn)
            if f not in LIST_FIELDS
            else int(a.tp[:, i].sum() + a.fn[:, i].sum()),
        }
    out["per_field"] = per_field
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        boots: dict[str, list[float]] = {k: [] for k in HEADLINE}
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            s = _summary(a, idx)
            for k in HEADLINE:
                boots[k].append(s[k])
        out["ci95"] = {
            k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in boots.items()
        }
        out["bootstrap"] = {"n_boot": n_boot, "seed": seed, "method": "percentile, resampling examples"}
    return out
