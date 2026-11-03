"""Automatic failure taxonomy for one scored example.

Categories (an example can have several):
  invalid_json        no parseable JSON object (and not a truncation)
  truncation          an object was started but never closed (usually max_new_tokens)
  schema_invalid      JSON parsed but violates the schema (extra keys, bad enum, min > max ...)
  wrong_salary_unit   salary off by a unit factor (x12 month/year, x1e5 lakh, x1000 k, x10/x100)
  salary_wrong_value  salary wrong for any other reason (missing, wrong bound...)
  missed_skill        a gold skill not predicted
  spurious_skill      a predicted skill not in gold
  hallucinated_field  a field that is null/[] in gold was filled in
  location_confusion  predicted locations differ from gold (missed/extra/HQ city)
  seniority_mismatch  seniority differs and both are stated
  experience_wrong    min/max experience wrong
  role_or_company_wrong
  other_field_wrong   work_mode / employment_type / education wrong
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from jobx.eval.metrics import ExampleScore
from jobx.normalize.salary import to_inr_per_annum

CATEGORIES: tuple[str, ...] = (
    "invalid_json",
    "truncation",
    "schema_invalid",
    "wrong_salary_unit",
    "salary_wrong_value",
    "missed_skill",
    "spurious_skill",
    "hallucinated_field",
    "location_confusion",
    "seniority_mismatch",
    "experience_wrong",
    "role_or_company_wrong",
    "other_field_wrong",
)
_UNIT_FACTORS = (12.0, 1 / 12, 1e5, 1e-5, 1e3, 1e-3, 10.0, 0.1, 100.0, 0.01, 1e5 / 12, 12 / 1e5)


def _salary_unit_error(gold: Any, pred: Any) -> bool:
    if not gold or not pred:
        return False
    for g, p in zip(gold, pred, strict=False):
        if g is None or p is None or g == 0:
            continue
        r = p / g
        if any(abs(r / f - 1) < 0.02 for f in _UNIT_FACTORS):
            return True
    return False


def classify(s: ExampleScore) -> list[str]:
    cats: list[str] = []
    if not s.json_ok:
        cats.append("truncation" if s.truncated else "invalid_json")
        return cats
    if not s.schema_ok:
        cats.append("schema_invalid")
    f = s.fields
    if not f["salary"].correct:
        g = to_inr_per_annum(s.gold.get("salary"))
        p = to_inr_per_annum(s.pred.get("salary"))
        if _salary_unit_error(g, p):
            cats.append("wrong_salary_unit")
        elif not f["salary"].hallucinated:
            cats.append("salary_wrong_value")
    if f["skills"].fn:
        cats.append("missed_skill")
    if f["skills"].fp:
        cats.append("spurious_skill")
    if any(fs.hallucinated for fs in f.values()):
        cats.append("hallucinated_field")
    if not f["locations"].correct and not f["locations"].hallucinated:
        cats.append("location_confusion")
    if not f["seniority"].correct and not f["seniority"].gold_null and not f["seniority"].pred_null:
        cats.append("seniority_mismatch")
    exp_wrong = not (f["min_experience_years"].correct and f["max_experience_years"].correct)
    exp_halluc = f["min_experience_years"].hallucinated or f["max_experience_years"].hallucinated
    if exp_wrong and not exp_halluc:
        cats.append("experience_wrong")
    if any(not f[k].correct and not f[k].hallucinated for k in ("role", "company")):
        cats.append("role_or_company_wrong")
    if any(not f[k].correct and not f[k].hallucinated for k in ("work_mode", "employment_type", "education")):
        cats.append("other_field_wrong")
    return cats


def taxonomy_counts(scores: Iterable[ExampleScore]) -> dict[str, int]:
    c: Counter[str] = Counter()
    n = 0
    for s in scores:
        n += 1
        c.update(classify(s))
    out = {k: c.get(k, 0) for k in CATEGORIES}
    out["n_examples"] = n
    return out


def worked_examples(
    scores: Sequence[ExampleScore], gold: dict[str, dict[str, Any]], category: str, k: int = 3
) -> list[dict[str, Any]]:
    out = []
    for s in scores:
        if category in classify(s):
            g = gold[s.id]
            diffs = {
                f: {"gold": s.gold_norm[f], "pred": s.pred_norm[f]}
                for f, fs in s.fields.items()
                if not fs.correct
            }
            out.append({"id": s.id, "text": g["text"], "diffs": diffs})
            if len(out) >= k:
                break
    return out
