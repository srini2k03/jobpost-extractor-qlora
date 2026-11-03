import json

from jobx.baselines.regex_extractor import extract, predict
from jobx.schema import JobPosting

POST = """🔥 Hiring!! Sr. Python Developer @ Kaveri Analytics Pvt Ltd
Exp: 4-7 yrs | CTC 18-24 LPA
Loc: Bangalore / Pune (Hybrid)
Must know Django, AWS, PostgreSQL, Docker
B.Tech/B.E. preferred. Call HR 9876543210. Joining kit worth ₹500!"""


def test_regex_baseline_extracts_core_fields():
    d = extract(POST)
    JobPosting.model_validate(d)
    assert d["company"] == "Kaveri Analytics"
    assert d["seniority"] == "senior"
    assert d["role"] == "Python Developer"
    assert d["min_experience_years"] == 4 and d["max_experience_years"] == 7
    assert d["salary"] == {"min": 1800000, "max": 2400000, "currency": "INR", "period": "year"}
    assert set(d["locations"]) == {"Bengaluru", "Pune"}
    assert set(d["skills"]) >= {"python", "django", "aws", "postgresql", "docker"}
    assert d["work_mode"] == "hybrid" and d["education"] == "b.tech"


def test_hq_city_not_a_location():
    d = extract("Role: Data Analyst\nLocation: Pune\nAcme Data is a SaaS company headquartered in Mumbai.")
    assert d["locations"] == ["Pune"]


def test_predict_is_json():
    assert json.loads(predict("nothing useful"))["salary"] is None
