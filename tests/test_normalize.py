import pytest

from jobx.normalize import find_experience, find_salaries, normalise_field, salary_key, to_inr_per_annum
from jobx.normalize.fields import norm_company, norm_role


@pytest.mark.parametrize(
    "text, expected",
    [
        ("₹12–18 LPA", (1_200_000, 1_800_000, "INR", "year")),
        ("CTC 8-10 lakhs", (800_000, 1_000_000, "INR", "year")),
        ("Salary: 1.2L/month", (120_000, 120_000, "INR", "month")),
        ("INR 12,00,000 - 18,00,000 p.a.", (1_200_000, 1_800_000, "INR", "year")),
        ("15k-25k p.m.", (15_000, 25_000, "INR", "month")),
        ("Rs. 12 to 18 Lakhs per annum", (1_200_000, 1_800_000, "INR", "year")),
        ("₹15,000 - ₹25,000 per month", (15_000, 25_000, "INR", "month")),
        ("₹ 12L-18L PA", (1_200_000, 1_800_000, "INR", "year")),
        ("upto 18 LPA", (None, 1_800_000, "INR", "year")),
        ("Min 12 LPA", (1_200_000, None, "INR", "year")),
        ("$40k-60k/yr", (40_000, 60_000, "USD", "year")),
        ("Stipend: ₹10,000/month", (10_000, 10_000, "INR", "month")),
        ("Package 20 to 30 LPA tak hai", (2_000_000, 3_000_000, "INR", "year")),
    ],
)
def test_salary_formats(text, expected):
    ms = find_salaries(text)
    assert ms, text
    m = ms[0]
    assert (m.min, m.max, m.currency, m.period) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Joining kit worth ₹500 for all new joiners",
        "Call HR: +91 9876543210",
        "Walk-in 10 AM - 5 PM",
        "Batch 2024 passouts, 3-5 years",
        "Notice period: 30 days",
    ],
)
def test_salary_distractors_rejected(text):
    assert find_salaries(text) == []


def test_salary_normalisation_equivalences():
    month = {"min": 100000, "max": 150000, "currency": "INR", "period": "month"}
    year = {"min": 1200000, "max": 1800000, "currency": "INR", "period": "year"}
    assert to_inr_per_annum(month) == (1_200_000, 1_800_000)
    assert salary_key(month) == salary_key(year) == salary_key("12-18 LPA")
    assert salary_key({"min": 20000, "max": 30000, "currency": "USD", "period": "year"}) == (1660, 2490)
    assert salary_key(None) is None
    assert salary_key({"min": None, "max": None}) is None
    assert salary_key({"min": "12 LPA", "max": "18 LPA", "period": "year"}) == (1200, 1800)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Exp: 3-5 yrs", (3, 5)),
        ("5+ years", (5, None)),
        ("Freshers can apply", (0, 0)),
        ("minimum 2 years", (2, None)),
        ("three to five years", (3, 5)),
        ("upto 6 Yrs", (None, 6)),
        ("Eligibility: 2 – 4 Yrs", (2, 4)),
    ],
)
def test_experience(text, expected):
    m = find_experience(text)[0]
    assert (m.min, m.max) == expected


def test_experience_ignores_business_age_and_freshers_and_experienced():
    assert find_experience("We have been in business for over 15 years.") == []
    assert find_experience("Freshers and experienced both can apply") == []


def test_skill_and_location_canonicalisation():
    assert normalise_field("skills", ["ReactJS", "K8s", "node js", "PowerBI", "unknown thing"]) == tuple(
        sorted(["react", "kubernetes", "node.js", "power bi", "unknown thing"])
    )
    assert normalise_field("locations", ["Bangalore", "Gurgaon", "Bombay"]) == (
        "Bengaluru",
        "Gurugram",
        "Mumbai",
    )
    assert normalise_field("skills", "Python, SQL") == ("python", "sql")
    assert normalise_field("skills", None) == ()


def test_enums_numbers_company_role():
    assert normalise_field("work_mode", "Work from Home") == "remote"
    assert normalise_field("employment_type", "Full Time") == "full_time"
    assert normalise_field("education", "B.E.") == "b.tech"
    assert normalise_field("min_experience_years", "3 years") == 3.0
    assert normalise_field("min_experience_years", "two") == 2.0
    assert (
        norm_company("Kaveri Analytics Pvt. Ltd.") == norm_company("kaveri analytics") == "kaveri analytics"
    )
    assert norm_role("Sr. Data Engineer") == norm_role("Data Engineer (Senior)") == "data engineer"
    assert norm_role(None) is None
