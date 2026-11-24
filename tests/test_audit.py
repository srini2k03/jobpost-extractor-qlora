import json

from conftest import ROOT
from jobx.data.audit import GATES, audit, run_audit


def _ex(i, text, family, company="Acme"):
    return {
        "id": f"x{i}",
        "text": text,
        "label": {"company": company, "role": "Dev", "skills": ["python"]},
        "meta": {"family": family, "template": f"{family}/v1"},
    }


BASE = (
    "Hiring Python developer at Acme in Pune with Django SQL and AWS experience three to five years "
    "salary twelve LPA apply by email to careers team before the end of the month"
)


def test_committed_splits_pass_every_gate():
    rep = run_audit(ROOT, out=None)
    assert rep["ok"], rep["gates"]
    assert set(rep["gates"]) == set(GATES)
    assert rep["few_shot_exemplars"] and set(rep["few_shot_exemplars"].values()) == {"train"}


def test_committed_audit_report_is_current():
    committed = json.loads((ROOT / "results/audit/leakage.json").read_text())
    fresh = run_audit(ROOT, out=None)
    for k in ("n", "exact_duplicates", "label_overlap", "metadata_overlap", "gates", "ok"):
        assert committed[k] == fresh[k], k


def test_audit_catches_a_near_duplicate_and_shared_metadata():
    near = BASE.replace("month", "week")
    splits = {
        "train": [
            _ex(1, BASE, "fam_a"),
            _ex(2, "completely different text about accounting jobs in Delhi " * 3, "fam_a"),
        ],
        "test_synth": [_ex(3, near, "fam_a", company="Acme")],
    }
    rep = audit(splits)
    assert rep["near_duplicates"]["cross_split_pairs_at"]["0.6"] == 1
    assert rep["label_overlap"]["train~test_synth"]["identical_records"] == 1
    assert rep["metadata_overlap"]["train~test_synth"]["families"] == ["fam_a"]
    assert not rep["ok"]


def test_audit_catches_exact_duplicates_after_normalisation():
    splits = {"train": [_ex(1, BASE, "a")], "val": [_ex(2, BASE.upper() + "!!", "b", company="Other")]}
    rep = audit(splits)
    assert rep["exact_duplicates"]["raw"]["cross_split"] == 0
    assert rep["exact_duplicates"]["normalised"]["cross_split"] == 1
    assert not rep["gates"]["max_cross_split_exact_duplicates"]["ok"]
