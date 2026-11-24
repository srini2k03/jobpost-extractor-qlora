import pytest
from pydantic import ValidationError

from conftest import gold_example
from jobx.schema import JobPosting
from jobx.schema.v2 import JobPostingV2, from_v1


def test_v1_to_v2_keeps_every_value_and_invents_nothing():
    v1 = JobPosting.model_validate(gold_example())
    v2 = from_v1(v1)
    assert v2.company == v1.company and v2.role == v1.role and v2.required_skills == v1.skills
    assert [loc.city for loc in v2.locations] == v1.locations and all(
        loc.country == "IN" for loc in v2.locations
    )
    assert v2.salary is not None and v2.salary.min == v1.salary.min and v2.salary.period == v1.salary.period
    assert (
        v2.preferred_skills == []
        and v2.application_deadline is None
        and v2.contact.emails == []
        and v2.openings is None
    )
    assert v2.schema_version == "2.0-draft"


def test_v2_validation_rules():
    JobPostingV2(required_skills=["python"], preferred_skills=["go"], application_deadline="2026-12-31")
    with pytest.raises(ValidationError):
        JobPostingV2(required_skills=["python"], preferred_skills=["python"])
    with pytest.raises(ValidationError):
        JobPostingV2(min_experience_years=5, max_experience_years=2)
    with pytest.raises(ValidationError):
        JobPostingV2(eligibility={"min_age": 30, "max_age": 21})
    with pytest.raises(ValidationError):
        JobPostingV2(unknown_field=1)
