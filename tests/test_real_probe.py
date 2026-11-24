from jobx.data.real_probe import (
    build_pool,
    card_licence,
    excerpt,
    guess_text_column,
    is_job_post,
    licence_from_tags,
    match_job_domain,
    row_stats,
)

AD = (
    "Job Title: Python Developer\nLocation: Pune\nExperience: 3-5 years\nSalary: 8-12 LPA\n"
    "Skills required: Python, Django, SQL\nResponsibilities: build APIs.\nApply before 30 June."
)


def test_match_job_domain_suffix_only():
    assert match_job_domain("https://www.naukri.com/job-listings-python-developer-123") == "naukri.com"
    assert match_job_domain("https://in.indeed.com/viewjob?jk=1") == "in.indeed.com"
    assert match_job_domain("https://notnaukri.com/x") is None
    assert match_job_domain("https://example.com/naukri.com") is None
    assert match_job_domain("not a url") is None


def test_is_job_post_accepts_an_ad_and_rejects_prose_and_listings():
    assert is_job_post(AD)
    assert not is_job_post("A blog post about the weather in Pune and the monsoon.")
    listing = AD + "\n" + "\n".join(["Data Analyst - apply now - posted 3 days ago"] * 5)
    assert not is_job_post(listing)


def test_build_pool_filters_dedups_caps_and_is_deterministic():
    rows = [{"text": AD, "domain": "naukri.com", "doc_id": "a"}]
    rows += [{"text": AD + " ", "domain": "naukri.com", "doc_id": "b"}]  # same after normalisation
    rows += [
        {"text": AD.replace("Pune", f"Pune {i}"), "domain": "shine.com", "doc_id": f"s{i}"} for i in range(5)
    ]
    rows += [{"text": "too short", "domain": "shine.com", "doc_id": "x"}]
    pool, stats = build_pool(rows, n_pool=10, seed=1, min_chars=50, max_chars=1000, max_per_domain=3)
    assert stats["dropped"] == {"exact_duplicate": 1, "length": 1}
    assert stats["pool_per_domain"] == {"shine.com": 3, "naukri.com": 1}
    again, _ = build_pool(rows, n_pool=10, seed=1, min_chars=50, max_chars=1000, max_per_domain=3)
    assert [r["doc_id"] for r in pool] == [r["doc_id"] for r in again]


def test_licence_helpers_and_text_column():
    assert card_licence({"license": ["CC-BY-4.0", "mit"]}) == "cc-by-4.0"
    assert card_licence(None) is None
    assert licence_from_tags(["task_categories:other", "license:odc-by"]) == "odc-by"
    rows = [{"title": "Dev", "description": AD, "n": 1}, {"title": "QA", "description": AD * 2, "n": 2}]
    assert guess_text_column(rows) == "description"
    st = row_stats(rows, "description", min_chars=100, max_chars=10_000)
    assert st["rows_read"] == 2 and st["indian_in_length_range"] == 2
    assert excerpt("a  b\n c", 3) == "a b…"


def test_india_row_prefers_location_columns():
    from jobx.data.real_probe import india_row

    assert india_row({"country": "India", "d": "x"}, "d")
    assert india_row({"location": "IN, MH, Mumbai", "d": "x"}, "d")
    assert india_row({"job_location": "Bengaluru, Karnataka", "d": "x"}, "d")
    assert not india_row({"location": "Indiana, US", "d": "salary 12 LPA"}, "d")
    assert india_row({"d": "Hiring in Pune, 12 LPA"}, "d")
