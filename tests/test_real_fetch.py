from jobx.data.real_fetch import compose, is_indian, select

EMS = {
    "title": "Data Analyst",
    "location": "IN, MH, Mumbai",
    "employment_type": "Full-time",
    "salary_range": None,
    "description": "We need an analyst.",
    "requirements": "2+ years SQL",
    "benefits": "",
    "fraudulent": 0,
}


def test_compose_uses_only_real_field_values():
    t = compose("emscad", EMS)
    assert t.startswith("Data Analyst\nLocation: IN, MH, Mumbai\nEmployment type: Full-time\n\n")
    assert "Salary" not in t and t.endswith("We need an analyst.\n\n2+ years SQL")
    j = compose(
        "joa",
        {"title": "SRE", "employer": "Acme", "location": "Pune", "description": "Run things.\r\n\n\n\nNow"},
    )
    assert j == "SRE\nAcme · Pune\n\nRun things.\n\nNow"


def test_india_filter_does_not_confuse_indiana():
    assert is_indian("emscad", EMS)
    assert not is_indian("emscad", {**EMS, "location": "US, IN, Indianapolis"})
    assert not is_indian("emscad", {**EMS, "fraudulent": 1})
    assert is_indian("joa", {"country": "IN"}) and not is_indian(
        "joa", {"country": "US", "location": "Fort Wayne, IN"}
    )


def test_select_filters_dedups_and_samples_deterministically():
    rows = [
        {**EMS, "job_id": str(i), "description": f"We need an analyst number {i}. " * 5} for i in range(10)
    ]
    rows += [{**EMS, "job_id": "dup", "description": rows[0]["description"]}]
    rows += [{**EMS, "job_id": "us", "location": "US, NY, New York"}]
    src = {"kind": "emscad", "n": 4}
    a, c = select(rows, src, min_chars=50, max_chars=5000, seed=3)
    b, _ = select(rows, src, min_chars=50, max_chars=5000, seed=3)
    assert [r["source_row"] for r in a] == [r["source_row"] for r in b] and len(a) == 4
    assert c["india"] == 11 and c["duplicate"] == 1 and c["eligible"] == 10
