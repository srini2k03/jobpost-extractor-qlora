import json
from pathlib import Path

from jobx.data.dedup import MinHasher, dedup, jaccard, shingles
from jobx.data.split import SplitSpec, leakage_report


def test_minhash_estimates_jaccard():
    a = shingles("the quick brown fox jumps over the lazy dog " * 3 + "alpha beta gamma delta")
    b = shingles("the quick brown fox jumps over the lazy dog " * 3 + "alpha beta gamma omega")
    mh = MinHasher(256)
    est = float((mh.signature(a) == mh.signature(b)).mean())
    assert abs(est - jaccard(a, b)) < 0.15


def test_dedup_removes_near_and_exact_and_respects_priority():
    base = "Senior Data Engineer at Kaveri Analytics. Exp 5-8 yrs. CTC 18-24 LPA. Bangalore. Python Spark AWS Kafka Airflow."
    texts = [
        base,  # 0 train
        base,  # 1 train, exact dup
        base.replace("Kafka", "Kafka,"),  # 2 test (punctuation-only change -> exact after normalising)
        base.replace("Airflow.", "Airflow and Snowflake."),  # 3 near dup
        "Completely different posting for a Telecaller in Chennai, 15k-20k per month, freshers.",
    ]
    prio = [2, 2, 0, 2, 2]
    r = dedup(texts, priority=prio, threshold=0.7)
    assert 2 in r.keep, "test copy must survive"
    assert 0 not in r.keep and 1 not in r.keep
    assert 3 not in r.keep and 4 in r.keep


def test_split_assignment_rules():
    spec = SplitSpec(
        families={"train": ["a"], "val": ["b"], "test_synth": ["c"]},
        company_buckets={"train": (0, 70), "val": (70, 85), "test_synth": (85, 100)},
    )
    assert spec.assign("paraphrase/a", None) == "train"
    comp_train = next(f"Co {i}" for i in range(500) if spec.company_split(f"Co {i}") == "train")
    comp_test = next(f"Co {i}" for i in range(500) if spec.company_split(f"Co {i}") == "test_synth")
    assert spec.assign("a", comp_train) == "train"
    assert spec.assign("a", comp_test) is None
    assert spec.assign("c", comp_test) == "test_synth"


def test_committed_dataset_is_leakage_free_and_matches_stats():
    from jobx.data.build import read_jsonl

    splits = {s: read_jsonl(f"data/processed/{s}.jsonl") for s in ("train", "val", "test_synth")}
    rep = leakage_report(splits)
    assert rep["ok"], rep
    stats = json.loads(Path("data/processed/stats.json").read_text())
    assert stats["pipeline"]["final"] == {s: len(v) for s, v in splits.items()}
    ids = [e["id"] for v in splits.values() for e in v]
    assert len(ids) == len(set(ids))


def test_datacard_is_current():
    from jobx.data.datacard import render

    stats = json.loads(Path("data/processed/stats.json").read_text())
    assert Path("DATA_CARD.md").read_text() == render(stats, None), "run `jobx datacard`"
