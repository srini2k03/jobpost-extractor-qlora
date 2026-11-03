import json
import math

import pytest

from conftest import gold_example
from jobx.eval.metrics import aggregate, score_example, score_field
from jobx.eval.taxonomy import classify, taxonomy_counts


def out(d: dict) -> str:
    return json.dumps(d)


def test_perfect_prediction():
    g = gold_example()
    s = score_example("x", g, out(g))
    agg = aggregate([s], n_boot=0)
    assert agg["micro_f1"] == 1.0 and agg["macro_f1"] == 1.0 and agg["field_em_mean"] == 1.0
    assert (
        agg["hallucination_rate"] != agg["hallucination_rate"] or agg["hallucination_rate"] == 0
    )  # nan: no nulls


def test_equivalent_surface_forms_count_as_correct():
    g = gold_example()
    pred = {
        **g,
        "company": "Kaveri Analytics Pvt. Ltd.",
        "role": "Sr. Data Engineer",
        "skills": ["Python", "PySpark", "Spark", "AWS"][::-1],
        "salary": {"min": 150000, "max": 200000, "currency": "INR", "period": "month"},
        "locations": ["Bangalore", "Poona"],
        "work_mode": "hybrid",
        "education": "b.tech",
    }
    pred["skills"] = ["Python", "Spark", "Amazon Web Services"]
    s = score_example("x", g, out(pred))
    assert all(f.correct for f in s.fields.values()), {k: (s.gold_norm[k], s.pred_norm[k]) for k in s.fields}


def test_list_prf_and_hallucination():
    g = {**gold_example(), "education": None, "work_mode": None}
    p = {**g, "skills": ["python", "docker"], "work_mode": "remote"}
    s = score_example("x", g, out(p))
    sk = s.fields["skills"]
    assert (sk.tp, sk.fp, sk.fn) == (1, 1, 2)
    assert s.fields["work_mode"].hallucinated and not s.fields["education"].hallucinated
    agg = aggregate([s], n_boot=0)
    assert agg["hallucination_rate"] == pytest.approx(0.5)  # 1 of 2 gold-null fields filled
    assert agg["per_field"]["skills"]["precision"] == pytest.approx(0.5)
    assert agg["per_field"]["skills"]["recall"] == pytest.approx(1 / 3)


def test_invalid_json_scores_as_all_missing():
    g = gold_example()
    s = score_example("x", g, "I cannot help with that")
    assert not s.json_ok
    agg = aggregate([s], n_boot=0)
    assert agg["json_validity"] == 0 and agg["micro_recall"] == 0
    assert classify(s) == ["invalid_json"]


def test_scalar_counts():
    f = score_field("company", "Acme", "Other")
    assert (f.tp, f.fp, f.fn, f.correct) == (0, 1, 1, False)
    f = score_field("company", None, None)
    assert (f.tp, f.fp, f.fn, f.correct) == (0, 0, 0, True)
    f = score_field("salary", None, {"min": 1, "max": 2})
    assert f.hallucinated


def test_bootstrap_ci_brackets_point_estimate():
    g = gold_example()
    scores = []
    for i in range(60):
        p = dict(g) if i % 3 else {**g, "company": None, "skills": ["python"]}
        scores.append(score_example(str(i), g, out(p)))
    agg = aggregate(scores, n_boot=300, seed=1)
    lo, hi = agg["ci95"]["micro_f1"]
    assert lo <= agg["micro_f1"] <= hi and hi - lo > 0
    assert math.isclose(agg["field_em_mean"], sum(v["em"] for v in agg["per_field"].values()) / 11)


def test_taxonomy_categories():
    g = gold_example()
    unit = score_example(
        "u", g, out({**g, "salary": {"min": 18, "max": 24, "currency": "INR", "period": "year"}})
    )
    assert "wrong_salary_unit" in classify(unit)
    lakh = score_example(
        "l", g, out({**g, "salary": {"min": 1800000, "max": 2400000, "currency": "INR", "period": "month"}})
    )
    assert "wrong_salary_unit" in classify(lakh)
    trunc = score_example("t", g, out(g)[:80])
    assert classify(trunc) == ["truncation"]
    loc = score_example("c", g, out({**g, "locations": ["Bengaluru", "Mumbai"]}))
    assert "location_confusion" in classify(loc)
    sen = score_example("s", g, out({**g, "seniority": "lead"}))
    assert "seniority_mismatch" in classify(sen)
    hall = score_example("h", {**g, "education": None}, out(g))
    assert "hallucinated_field" in classify(hall)
    counts = taxonomy_counts([unit, trunc, loc])
    assert counts["n_examples"] == 3 and counts["truncation"] == 1
