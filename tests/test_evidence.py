import json

import pytest

from conftest import ROOT
from jobx.evidence import evidence_by_field, find_evidence, find_fuzzy, missing_cues, summarise

TEXT = """🔥 Hiring!! Sr. Python Developer @ Kaveri Analytics Pvt Ltd
Exp: 4-7 yrs | CTC 18-24 LPA
Loc: Bangalore / Pune (Hybrid)
Must know Django, AWS, PostgreSQL, Docker
B.Tech/B.E. preferred. Call HR 9876543210. Joining kit worth ₹500!"""
REC = {
    "company": "Kaveri Analytics",
    "role": "Python Developer",
    "seniority": "senior",
    "skills": ["python", "django", "aws", "postgresql", "docker"],
    "min_experience_years": 4,
    "max_experience_years": 7,
    "salary": {"min": 1800000, "max": 2400000, "currency": "INR", "period": "year"},
    "locations": ["Bengaluru", "Pune"],
    "work_mode": "hybrid",
    "employment_type": None,
    "education": "b.tech",
}


def _by(evs):
    return {(e.field, json.dumps(e.value, sort_keys=True)): e for e in evs}


def test_every_value_of_the_readme_example_is_supported_with_real_spans():
    evs = find_evidence(TEXT, REC)
    assert len(evs) == 15 and all(e.status == "supported" for e in evs)
    for e in evs:
        assert e.spans and all(TEXT[s.start : s.end] == s.text for s in e.spans)
    by = _by(evs)
    assert by[("locations", '"Bengaluru"')].match == "alias"
    assert by[("locations", '"Bengaluru"')].spans[0].text == "Bangalore"
    assert by[("salary", json.dumps(REC["salary"], sort_keys=True))].spans[0].text == "18-24 LPA"
    assert by[("min_experience_years", "4.0")].match == "parsed"
    assert summarise(evs)["supported_share"] == 1.0


def test_values_not_in_the_text_are_flagged_unsupported():
    rec = {
        **REC,
        "skills": ["python", "kubernetes"],
        "company": "Infosys",
        "work_mode": "remote",
        "salary": {"min": 3000000, "max": 4000000, "currency": "INR", "period": "year"},
    }
    by = _by(find_evidence(TEXT, rec))
    assert by[("skills", '"kubernetes"')].status == "unsupported"
    assert by[("skills", '"kubernetes"')].spans == []
    assert by[("company", '"Infosys"')].status == "unsupported"
    assert by[("work_mode", '"remote"')].status == "unsupported"
    assert by[("salary", json.dumps(rec["salary"], sort_keys=True))].status == "unsupported"
    assert evidence_by_field(find_evidence(TEXT, rec))["skills"] == "unsupported"


def test_partial_salary_and_experience_are_weak():
    rec = {
        **REC,
        "salary": {"min": 1800000, "max": 3000000, "currency": "INR", "period": "year"},
        "min_experience_years": 4,
        "max_experience_years": 9,
    }
    by = _by(find_evidence(TEXT, rec))
    assert by[("salary", json.dumps(rec["salary"], sort_keys=True))].match == "partial"
    assert by[("min_experience_years", "4.0")].status == "weak"
    assert by[("max_experience_years", "9.0")].status == "unsupported"


def test_normalised_and_fuzzy_matches():
    text = "Company: Kaveri Analytics Pvt. Ltd.\nRole: Sr. Data Engineer\nCompny: Brightwave Technolgies"
    by = _by(find_evidence(text, {"company": "Kaveri Analytics Private Limited", "role": "Data Engineer"}))
    assert by[("company", '"Kaveri Analytics Private Limited"')].match == "normalised"
    assert by[("role", '"Data Engineer"')].match == "exact"
    span, ratio = find_fuzzy(text, "Brightwave Technologies")
    assert span is not None and span.text == "Brightwave Technolgies" and ratio >= 0.85
    by = _by(find_evidence(text, {"company": "Brightwave Technologies"}))
    assert by[("company", '"Brightwave Technologies"')].status == "weak"


def test_missing_cues_point_at_text_for_null_fields():
    cues = missing_cues(
        TEXT,
        {**REC, "salary": None, "locations": [], "min_experience_years": None, "max_experience_years": None},
    )
    assert cues["salary"].text == "18-24 LPA"
    assert cues["experience"].text == "4-7 yrs"
    assert cues["locations"].text in ("Bangalore", "Pune")
    assert "salary" not in missing_cues(TEXT, REC)


@pytest.mark.parametrize("split", ["test_synth"])
def test_gold_labels_are_almost_always_supported(split):
    rows = [json.loads(x) for x in (ROOT / f"data/processed/{split}.jsonl").read_text().splitlines()]
    evs = [e for r in rows for e in find_evidence(r["text"], r["label"])]
    s = summarise(evs)
    assert s["supported_share"] >= 0.998, s
