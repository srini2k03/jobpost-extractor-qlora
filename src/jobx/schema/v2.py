"""Schema v2: a richer target, DESIGNED but NOT trained (see docs/schema.md for the status and the plan).

The released model emits schema v1 (:class:`jobx.schema.models.JobPosting`). v2 adds fields recruiters ask for
(required vs preferred skills, a structured location with remote flag, application deadline, contact
channels, eligibility, openings) and keeps every v1 value. :func:`from_v1` converts a v1 record without
inventing anything: fields v1 cannot express stay null/empty. Training a v2 model needs labelled data for the
new fields (generator templates + real annotations) and a new CPU fine-tune (~2.5 h per run on the current
runner); that has not been done.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jobx.schema.models import Currency, Education, EmploymentType, JobPosting, Period, Seniority, WorkMode

SCHEMA_VERSION: Literal["2.0-draft"] = "2.0-draft"


class Location(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city: str | None = Field(default=None, description="Canonical city name (Bengaluru, Gurugram, ...)")
    state: str | None = None
    country: str | None = Field(default=None, description="ISO 3166-1 alpha-2, e.g. IN")


class SalaryV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: float | None = Field(default=None, ge=0)
    max: float | None = Field(default=None, ge=0)
    currency: Currency = "INR"
    period: Period = "year"
    variable_component: bool | None = Field(
        default=None, description="Incentives/bonus mentioned as part of pay"
    )

    @model_validator(mode="after")
    def _check(self) -> SalaryV2:
        if self.min is None and self.max is None:
            raise ValueError("salary needs min or max; use null for the whole salary")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("salary.min must be <= salary.max")
        return self


class Contact(BaseModel):
    """Application channels as written in the posting. Personal data: never logged, never used for training
    without consent; the API can be configured to drop this block."""

    model_config = ConfigDict(extra="forbid")

    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)


class Eligibility(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graduation_years: list[int] = Field(default_factory=list, description="e.g. 2024/2025 batch")
    min_age: int | None = Field(default=None, ge=14, le=80)
    max_age: int | None = Field(default=None, ge=14, le=80)
    other: list[str] = Field(default_factory=list, description="e.g. 'no backlogs', '60% throughout'")


class JobPostingV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0-draft"] = SCHEMA_VERSION
    company: str | None = None
    title: str | None = Field(default=None, description="Job title as stated (seniority words kept)")
    role: str | None = Field(default=None, description="v1 role: title without seniority words")
    seniority: Seniority | None = None
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    min_experience_years: float | None = Field(default=None, ge=0, le=50)
    max_experience_years: float | None = Field(default=None, ge=0, le=50)
    salary: SalaryV2 | None = None
    locations: list[Location] = Field(default_factory=list)
    remote_allowed: bool | None = None
    work_mode: WorkMode | None = None
    employment_type: EmploymentType | None = None
    education: list[Education] = Field(default_factory=list, description="All acceptable degrees")
    openings: int | None = Field(default=None, ge=1)
    application_deadline: date | None = None
    contact: Contact = Field(default_factory=Contact)
    eligibility: Eligibility = Field(default_factory=Eligibility)

    @model_validator(mode="after")
    def _check(self) -> JobPostingV2:
        lo, hi = self.min_experience_years, self.max_experience_years
        if lo is not None and hi is not None and lo > hi:
            raise ValueError("min_experience_years must be <= max_experience_years")
        e = self.eligibility
        if e.min_age is not None and e.max_age is not None and e.min_age > e.max_age:
            raise ValueError("eligibility.min_age must be <= max_age")
        overlap = set(self.required_skills) & set(self.preferred_skills)
        if overlap:
            raise ValueError(f"skills both required and preferred: {sorted(overlap)}")
        return self


def from_v1(p: JobPosting) -> JobPostingV2:
    """Lossless v1 -> v2. v1 does not separate required from preferred skills, so all go to required_skills;
    fields v1 has no counterpart for stay empty (nothing is inferred)."""
    sal = None
    if p.salary is not None:
        sal = SalaryV2(min=p.salary.min, max=p.salary.max, currency=p.salary.currency, period=p.salary.period)
    return JobPostingV2(
        company=p.company,
        role=p.role,
        seniority=p.seniority,
        required_skills=list(p.skills),
        min_experience_years=p.min_experience_years,
        max_experience_years=p.max_experience_years,
        salary=sal,
        locations=[Location(city=c, country="IN") for c in p.locations],
        remote_allowed=True if p.work_mode in ("remote", "hybrid") else None,
        work_mode=p.work_mode,
        employment_type=p.employment_type,
        education=[p.education] if p.education else [],
    )
