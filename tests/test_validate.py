from conftest import gold_example
from jobx.data.paraphrase import filter_paraphrases
from jobx.data.validate import check_no_new_facts, check_recoverable, hypotheses, validate_paraphrase

SRC = (
    "Senior Data Engineer at Kaveri Analytics Pvt Ltd\nExp: 5-8 yrs | CTC 18-24 LPA\n"
    "Location: Bangalore / Pune (Hybrid)\nSkills: Python, Spark, AWS\nB.Tech required. Full-time."
)


def test_source_recoverable():
    assert check_recoverable(SRC, gold_example()).ok


def test_paraphrase_dropping_salary_fails():
    para = SRC.replace("CTC 18-24 LPA", "great pay")
    res = validate_paraphrase(SRC, para, gold_example())
    assert not res.ok and "salary" in res.failures


def test_paraphrase_new_city_and_changed_salary_fail():
    para = SRC.replace("Bangalore / Pune", "Bangalore / Pune / Chennai").replace("18-24 LPA", "18-25 LPA")
    res = check_no_new_facts(SRC, para)
    assert not res.ok
    assert any(f.startswith("new:locations") for f in res.failures)
    assert any(f.startswith("new:salary") for f in res.failures)


def test_good_paraphrase_passes_with_nli_hook():
    para = (
        "Kaveri Analytics is hiring a Sr. Data Engineer in Bengaluru and Pune, hybrid. 5 to 8 years, "
        "Python, Spark and AWS. ₹18–24 LPA. B.E/B.Tech. Full Time role."
    )
    seen = []

    def fake_nli(premise: str, hyp: str) -> float:
        seen.append(hyp)
        return 0.9

    res = validate_paraphrase(SRC, para, gold_example(), nli=fake_nli)
    assert res.ok, res.failures
    assert len(seen) == len(hypotheses(gold_example()))
    res2 = validate_paraphrase(SRC, para, gold_example(), nli=lambda p, h: 0.1)
    assert not res2.ok and all(f.startswith("nli:") for f in res2.failures)


def test_filter_paraphrases_stats():
    src = {"id": "syn-1", "text": SRC, "label": gold_example(), "meta": {"family": "bullet_minimal"}}
    good = "Kaveri Analytics - Senior Data Engineer, 5-8 yrs, 18-24 LPA, Bangalore/Pune, Hybrid. Python, Spark, AWS. B.Tech. Full-time"
    kept, stats = filter_paraphrases(
        [(src, "terse", good), (src, "terse", "short"), (src, "terse", SRC + " Mumbai")]
    )
    assert len(kept) == 1 and kept[0]["meta"]["family"] == "paraphrase/bullet_minimal"
    assert stats["attempted"] == 3 and stats["kept"] == 1
    assert stats["reasons"]["empty_or_too_short"] == 1
