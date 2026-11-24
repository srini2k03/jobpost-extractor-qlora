# Output schema

The model returns one JSON object per posting. The schema is defined once in Pydantic
([`src/jobx/schema/models.py`](../src/jobx/schema/models.py)) and exported as JSON Schema
([`schema/jobposting.schema.json`](../schema/jobposting.schema.json), `jobx schema`). The same definition is used
for training targets, constrained decoding (llama.cpp JSON-schema grammar), validation in the API and the
browser, and scoring.

## Schema v1 (trained, released)

| Field | Type | Rules |
|---|---|---|
| `company` | string or null | Hiring company as named; not a recruitment agency; legal suffixes are ignored when scoring. |
| `role` | string or null | Job title without seniority words ("Sr. Data Engineer" → "Data Engineer"). |
| `seniority` | `intern` `junior` `mid` `senior` `lead` `manager` or null | Only when stated in words. |
| `skills` | list of strings | Every skill, tool or language named (title included), lower-case canonical names (`vocab.SKILL_ALIASES`). |
| `min_experience_years`, `max_experience_years` | number 0–50 or null | "3-5 yrs" → 3/5, "5+ years" → 5/null, "freshers" → 0/0; min ≤ max. |
| `salary` | object or null | `{min, max, currency: INR or USD, period: year, month or hour}`, absolute amounts in the stated period ("12-18 LPA" → 1,200,000–1,800,000 per year; "1.2L/month" → 120,000 per month); at least one bound; min ≤ max. Bonuses, kits and fees are not salary. |
| `locations` | list of strings | Cities where the job is based, canonical names (Bangalore → Bengaluru); not head offices. |
| `work_mode` | `onsite` `hybrid` `remote` or null | Only when stated. |
| `employment_type` | `full_time` `part_time` `contract` `internship` `freelance` or null | Only when stated. |
| `education` | one of 15 degree values or null | `10th` `12th` `diploma` `graduate` `b.tech` `bca` `b.sc` `b.com` `bba` `mca` `m.tech` `m.sc` `mba` `ca` `phd`. |

**Null convention.** A scalar the posting does not state is `null`; an empty list is `[]`. Every key is always
present (`extra="forbid"`, all 11 keys required by the grammar). "Not stated" and "stated as none" are not
distinguished in v1. Nothing is inferred: a salary range is never estimated, a city is never guessed from a
phone code.

**Ambiguous cases** (conventions used for training data, the robustness suite and the real-posting labels):

| Situation | Convention |
|---|---|
| Several salary bands for one role ("6-8 LPA freshers, 10-14 LPA experienced") | overall range (6–14 LPA); each band is accepted as an alternative in the robustness suite |
| Headline "upto X" vs a labelled CTC line | the labelled line; the headline value accepted as an alternative |
| Monthly vs annual | keep the stated period; the evaluator compares INR per annum (12 months, 2,080 hours, 83 INR/USD) |
| No period, amount < ₹1 lakh / ≥ ₹1 lakh | monthly / yearly (the parser's rule; real-posting rows note when it was applied) |
| Several locations | all of them, in text order |
| Remote with an office city | `work_mode: remote`; the city is listed only if the job is based there |
| Per-skill experience that contradicts the labelled range | the labelled range; the other accepted as an alternative |
| No title / no company | `role: null` / `company: null` |
| Not a job posting at all | every field null / [] |
| Instructions inside the posting ("ignore previous instructions...") | treated as posting text; they never change the output |

**Validation.** `JobPosting.model_validate` rejects unknown keys, wrong types, out-of-range numbers, enums
outside the lists, min > max and a salary without bounds. The API returns these errors per field
(`POST /validate`); the browser demo mirrors the same checks in JavaScript.

## Evidence and confidence (output metadata, not schema fields)

`POST /extract` adds, next to `data`, the supporting span of every value (`evidence`: exact / alias / parsed /
normalised / partial / fuzzy / none; `jobx.evidence`), hints for possibly missed values, and calibrated field
confidence with optional abstention (`jobx.uncertainty`). See the README's API section.

## Schema v2 (designed, not trained)

[`src/jobx/schema/v2.py`](../src/jobx/schema/v2.py) defines `JobPostingV2` (Pydantic, validated, tested) with a
lossless `from_v1` converter. New fields:

| Field | Why |
|---|---|
| `title` (as written) next to `role` | keeps "Sr. Data Engineer" for display while `role` stays normalised |
| `required_skills` / `preferred_skills` | the most requested split; v1 merges them |
| `locations: [{city, state, country}]`, `remote_allowed` | real postings list states, countries and "remote (India)" |
| `salary.variable_component` | "8000 + incentives" is common in Indian postings |
| `education: [...]` | all acceptable degrees instead of the lowest one |
| `openings`, `application_deadline` | frequent in walk-in drives and government notifications |
| `contact {emails, phones, urls}` | application channels; personal data, so the API can drop the block and logs never contain it |
| `eligibility {graduation_years, min_age, max_age, other}` | campus and government hiring ("2024/2025 batch", "18-40 yrs") |

**Status and plan.** Not trained. A v2 model needs (1) generator templates and recoverability checks for the
new fields, (2) v2 labels for the real-posting set (the current labels are v1), (3) a new CPU fine-tune: about
2.5 h on one GitHub-hosted runner at the current data size, plus evaluation. `from_v1` lets v1 outputs be
served in the v2 shape meanwhile, with the new fields empty rather than guessed.
