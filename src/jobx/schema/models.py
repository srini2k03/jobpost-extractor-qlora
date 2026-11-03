"""The extraction target. One posting -> one ``JobPosting``.

Conventions (also spelled out in the system prompt):
* A scalar field is ``null`` when the posting does not state it. Never guess.
* List fields (``skills``, ``locations``) are ``[]`` when nothing is stated.
* ``salary`` amounts are absolute numbers in the stated period, e.g. "12-18 LPA" ->
  ``{"min": 1200000, "max": 1800000, "currency": "INR", "period": "year"}`` and
  "1.2L/month" -> ``{"min": 120000, "max": 120000, "currency": "INR", "period": "month"}``.
* ``role`` is the job title without seniority words ("Sr. Data Engineer" -> "Data Engineer");
  seniority goes to ``seniority`` only when the posting states it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Seniority = Literal["intern", "junior", "mid", "senior", "lead", "manager"]
WorkMode = Literal["onsite", "hybrid", "remote"]
EmploymentType = Literal["full_time", "part_time", "contract", "internship", "freelance"]
Education = Literal[
    "10th",
    "12th",
    "diploma",
    "graduate",
    "b.tech",
    "bca",
    "b.sc",
    "b.com",
    "bba",
    "mca",
    "m.tech",
    "m.sc",
    "mba",
    "ca",
    "phd",
]
Currency = Literal["INR", "USD"]
Period = Literal["year", "month", "hour"]

FIELD_ORDER: tuple[str, ...] = (
    "company",
    "role",
    "seniority",
    "skills",
    "min_experience_years",
    "max_experience_years",
    "salary",
    "locations",
    "work_mode",
    "employment_type",
    "education",
)
SCALAR_FIELDS: tuple[str, ...] = (
    "company",
    "role",
    "seniority",
    "min_experience_years",
    "max_experience_years",
    "salary",
    "work_mode",
    "employment_type",
    "education",
)
LIST_FIELDS: tuple[str, ...] = ("skills", "locations")


class Salary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: float | None = Field(
        default=None, ge=0, description="Lower bound, absolute amount in `period` units"
    )
    max: float | None = Field(
        default=None, ge=0, description="Upper bound, absolute amount in `period` units"
    )
    currency: Currency = "INR"
    period: Period = "year"

    @model_validator(mode="after")
    def _check(self) -> Salary:
        if self.min is None and self.max is None:
            raise ValueError("salary needs at least one of min/max; use null for the whole salary instead")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("salary.min must be <= salary.max")
        return self


class JobPosting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: str | None = Field(default=None, description="Hiring company (not a recruitment agency)")
    role: str | None = Field(default=None, description="Job title without seniority words")
    seniority: Seniority | None = None
    skills: list[str] = Field(default_factory=list, description="Required/preferred skills, canonical names")
    min_experience_years: float | None = Field(default=None, ge=0, le=50)
    max_experience_years: float | None = Field(default=None, ge=0, le=50)
    salary: Salary | None = None
    locations: list[str] = Field(default_factory=list, description="Indian cities, canonical names")
    work_mode: WorkMode | None = None
    employment_type: EmploymentType | None = None
    education: Education | None = None

    @model_validator(mode="after")
    def _check_exp(self) -> JobPosting:
        lo, hi = self.min_experience_years, self.max_experience_years
        if lo is not None and hi is not None and lo > hi:
            raise ValueError("min_experience_years must be <= max_experience_years")
        return self

    def to_json_dict(self) -> dict[str, object]:
        """Canonical dict with keys in ``FIELD_ORDER`` (what the model is trained to emit)."""
        raw = self.model_dump(mode="json")
        out: dict[str, object] = {k: raw[k] for k in FIELD_ORDER}
        sal = out["salary"]
        if isinstance(sal, dict):
            out["salary"] = {k: _intify(sal[k]) for k in ("min", "max", "currency", "period")}
        for k in ("min_experience_years", "max_experience_years"):
            out[k] = _intify(out[k])
        return out


def _intify(v: object) -> object:
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def json_schema() -> dict[str, object]:
    """JSON Schema for ``JobPosting`` (draft 2020-12, as emitted by Pydantic)."""
    schema = JobPosting.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "JobPosting"
    return schema
