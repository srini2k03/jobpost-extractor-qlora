import json

from conftest import ROOT
from jobx.data.build import read_jsonl
from jobx.data.real_build import AI_METHOD, build_real, validate_annotation, verify_annotations
from jobx.evidence import find_evidence
from jobx.schema import FIELD_ORDER, JobPosting


def _setup(tmp_path):
    (tmp_path / "data/real").mkdir(parents=True)
    pool = [
        {
            "id": "real-x-000",
            "text": "Python Developer\nAcme · Pune\n3-5 years Python, SQL",
            "source": {
                "dataset": "d/x",
                "revision": "abc",
                "licence": "cc0-1.0",
                "row_id": "7",
                "composition": "joa",
            },
        },
        {
            "id": "real-x-001",
            "text": "dup",
            "source": {"dataset": "d/x", "licence": "cc0-1.0", "composition": "joa"},
        },
    ]
    (tmp_path / "data/real/pool_hf.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool))
    anns = [
        {
            "id": "real-x-000",
            "label": {
                "company": "Acme",
                "role": "Python Developer",
                "skills": ["python", "sql"],
                "min_experience_years": 3,
                "max_experience_years": 5,
                "locations": ["Pune"],
            },
            "alt": {"role": ["Developer"]},
            "notes": "n",
        },
        {"id": "real-x-001", "reject": "near-duplicate"},
    ]
    (tmp_path / "data/real/annotations.jsonl").write_text("".join(json.dumps(a) + "\n" for a in anns))


def test_build_real_marks_ai_assisted_labels_and_keeps_provenance(tmp_path):
    _setup(tmp_path)
    meta = build_real(tmp_path, tmp_path / "data/real")
    rows = read_jsonl(tmp_path / "data/real/test_real.jsonl")
    assert meta["n"] == 1 and meta["rejected"] == 1 and meta["annotation"]["human_verified"] == 0
    r = rows[0]
    assert r["annotation"]["method"] == AI_METHOD and r["annotation"]["human_verified"] is False
    assert r["source"]["revision"] == "abc" and r["label_alternatives"] == {"role": ["Developer"]}
    assert list(r["label"]) == list(FIELD_ORDER)
    assert json.loads((tmp_path / "data/real/rejected.json").read_text()) == [
        {"id": "real-x-001", "reason": "near-duplicate"}
    ]


def test_verify_annotations_accept_edit_and_reject(tmp_path):
    _setup(tmp_path)
    answers = iter(["e", "skills", '["python"]', "a"])
    stats = verify_annotations(tmp_path, input_fn=lambda _: next(answers), verifier="tester")
    assert stats == {"accepted": 0, "edited": 1, "rejected": 0}
    ann = read_jsonl(tmp_path / "data/real/annotations.jsonl")[0]
    assert (
        ann["human_verified"] is True
        and ann["verified_by"] == "tester"
        and ann["label"]["skills"] == ["python"]
    )
    meta = build_real(tmp_path, tmp_path / "data/real")
    assert meta["annotation"]["human_verified"] == 1
    row = read_jsonl(tmp_path / "data/real/test_real.jsonl")[0]
    assert row["annotation"]["method"] == "human-verified"


def test_validate_annotation_checks_alternatives():
    a = validate_annotation({"id": "x", "label": {"role": "Dev"}, "alt": {"seniority": ["senior", None]}})
    assert a["alt"]["seniority"] == ["senior", None]
    try:
        validate_annotation({"id": "x", "label": {"role": "Dev"}, "alt": {"seniority": ["boss"]}})
    except Exception:
        pass
    else:
        raise AssertionError("invalid alternative accepted")


def test_committed_real_split_is_consistent():
    rows = read_jsonl(ROOT / "data/real/test_real.jsonl")
    meta = json.loads((ROOT / "data/real/real_meta.json").read_text())
    assert len(rows) == meta["n"] >= 100 and len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        JobPosting.model_validate(r["label"])
        assert r["annotation"]["human_verified"] or r["annotation"]["method"] == AI_METHOD
        assert r["source"].get("licence") or r["source"].get("corpus_licence")
    # gold values are mostly supported by the text (the rest are normalised/canonical forms)
    evs = [e for r in rows for e in find_evidence(r["text"], r["label"])]
    assert sum(e.status != "unsupported" for e in evs) / len(evs) > 0.8
    # rebuilding from the committed pools + annotations reproduces the committed file
    assert build_real(ROOT, ROOT / "data/real") == meta
