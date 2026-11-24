import json

from conftest import ROOT
from jobx.data.build import read_jsonl
from jobx.evidence import find_evidence
from jobx.robustness import CATEGORIES, PER_CATEGORY, build_suite, injection_succeeded, report
from jobx.schema import JobPosting


def test_suite_is_deterministic_and_matches_the_committed_file():
    rows = build_suite()
    assert len(rows) == len(CATEGORIES) * PER_CATEGORY
    assert rows == read_jsonl(ROOT / "data/robustness/robustness.jsonl")
    for r in rows:
        JobPosting.model_validate(r["label"])


def test_gold_is_grounded_and_injections_are_not_in_gold():
    for r in build_suite():
        cat = r["meta"]["category"]
        if cat == "irrelevant":
            assert all(v in (None, []) for v in r["label"].values())
            continue
        evs = find_evidence(r["text"], r["label"])
        assert all(e.status != "unsupported" for e in evs), (
            r["id"],
            [e.value for e in evs if e.status == "unsupported"],
        )
        if cat == "prompt_injection":
            assert r["meta"]["injection"] in r["text"]
            assert not injection_succeeded(r["label"], r["meta"]["injected"])


def test_report_scores_perfect_and_hijacked_predictions(tmp_path):
    rows = build_suite()
    good = tmp_path / "good__robustness.jsonl"
    good.write_text(
        "".join(json.dumps({"id": r["id"], "output": json.dumps(r["label"])}) + "\n" for r in rows)
    )
    rep = report(good, ROOT)
    assert set(rep) == set(CATEGORIES)
    assert all(v["micro_f1"] in (1.0, None) or v["micro_f1"] != v["micro_f1"] for v in rep.values())
    assert rep["prompt_injection"]["attack_success_rate"] == 0.0 and rep["irrelevant"]["all_null_rate"] == 1.0
    bad = tmp_path / "bad__robustness.jsonl"
    hijack = []
    for r in rows:
        lab = dict(r["label"])
        if r["meta"]["category"] == "prompt_injection":
            inj = r["meta"]["injected"]
            lab["company"] = inj.get("company", lab["company"])
            if "work_mode" in inj:
                lab["work_mode"] = inj["work_mode"]
            if "role" in inj:
                lab["role"] = inj["role"]
            if "location" in inj:
                lab["locations"] = [inj["location"]]
            if "skill" in inj:
                lab["skills"] = [inj["skill"]]
        hijack.append({"id": r["id"], "output": json.dumps(lab)})
    bad.write_text("".join(json.dumps(h) + "\n" for h in hijack))
    assert report(bad, ROOT)["prompt_injection"]["attack_success_rate"] == 1.0
