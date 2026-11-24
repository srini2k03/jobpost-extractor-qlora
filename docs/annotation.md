# Annotation guidelines and format (real-postings test set)

The real-world test split `data/real/test_real.jsonl` is built by `jobx real-build` from the fetched
pools (`data/real/pool_*.jsonl`) and the annotation file `data/real/annotations.jsonl`.

> **Status: AI-assisted annotation, not human-verified.** Every label in the current file was written
> with AI assistance in a single pass by reading the full posting text, following the rules below.
> No human has checked them yet. Each row of `test_real.jsonl` says so in
> `annotation.method` / `annotation.human_verified`. A person can verify them with `jobx verify-real`
> (see the end of this page); verified rows are then marked `human_verified: true` with the verifier's
> name and date, and `data/real/real_meta.json` counts them.

## Record format

`data/real/annotations.jsonl`, one JSON object per annotated pool posting:

```json
{"id": "real-joa-045",
 "label": {"company": "Binary Stream", "role": "Quality Assurance Analyst", "skills": ["manual testing", "sql"],
           "min_experience_years": 2, "max_experience_years": 4, "locations": ["Chandigarh"],
           "work_mode": "remote", "education": "graduate", "...": "every JobPosting key"},
 "alt": {"education": ["b.tech"], "role": ["QA Analyst"]},
 "notes": "'Location: Chandigarh (Currently Remote)'",
 "human_verified": false}
```

- `label` is a complete, schema-valid `JobPosting` (`src/jobx/schema/models.py`).
- `alt` lists **other acceptable values** for fields that the posting leaves genuinely ambiguous (a title
  that names two roles, an employment-type line that contradicts the body, an "In-office 4 days a week"
  work mode). The evaluator scores a prediction against the primary value and every alternative and keeps
  the most favourable match (`jobx.eval.metrics.best_field_score`). Synthetic splits have no alternatives.
- `notes` records why a decision was made (conflicts, conventions applied).
- `{"id": ..., "reject": "<reason>"}` removes a pool posting (not a single job ad, a near-duplicate of
  another selected posting, not India-based). Rejections are listed in `data/real/rejected.json`.

Each row of `test_real.jsonl` carries the posting `text`, the `label`, `label_alternatives`, the `source`
(dataset, pinned revision, licence, source row id, how the text was composed) and the `annotation` block.

## Field rules

The rules are the training conventions of the synthetic data (system prompt in `src/jobx/prompts.py`),
applied to real text. Anything the posting does not state is `null` (or `[]`); nothing is inferred.

| Field | Rule |
|---|---|
| `company` | The hiring company as named in the text or the source's employer line. An agency's unnamed client ("a US-based MNC") is `null`. |
| `role` | The job title without seniority words (senior, sr, junior, jr, trainee, intern, lead, team lead, mid, entry-level, II) and without location suffixes, requisition codes or "Hiring"/"Opening for". If the title is only a seniority word ("Interns"), `role` is `null`. |
| `seniority` | Only from an explicit level word: senior/sr → `senior`; junior/jr/trainee/entry-level → `junior`; lead/team lead/tech lead → `lead`; intern → `intern`; mid/II → `mid`; "managerial"/"manager level" → `manager`. A "Manager" job title is a role, not a level. Principal/Staff → `lead` with `null` accepted as an alternative. |
| `skills` | Skills, tools, technologies, languages, methods and domain skills the posting names as required or preferred, lower-cased, canonical alias names where `vocab.py` has one. Soft skills only when they are the vocabulary's skills (communication, negotiation, English/Hindi). Personality traits (self-motivated, attention to detail, problem-solving) and certifications are not skills. |
| `min/max_experience_years` | From explicit year counts: "3-5 years" → 3/5, "5+ years" or "minimum 5 years" → 5/null, "freshers" → 0/0. When the posting gives an overall and a per-skill figure, the overall one. Conflicting statements: the labelled line, the other as an alternative. |
| `salary` | Only explicit pay. Absolute amounts in the stated period; INR or USD. When the period is not stated, amounts below ₹1 lakh are read as monthly and larger ones as yearly (the parser's rule; noted per row). Bonuses, ESOPs and "competitive" are not a salary. |
| `locations` | Indian cities where the job is based, canonical names (Bangalore → Bengaluru, Gurgaon → Gurugram, Trivandrum → Thiruvananthapuram); cities outside the alias table keep their written name. Head offices, other offices, customer locations, "India" alone and states are not locations. |
| `work_mode` | Only when stated: remote / hybrid / onsite ("work from office", "in-office"). A temporary remote period or a "remote-first culture" statement about the company is not the role's work mode (accepted as an alternative). |
| `employment_type` | Only when stated (an "Employment type" line, "permanent", "contract", "internship", "freelance"). |
| `education` | The lowest acceptable degree when several are listed. A generic bachelor's ("Bachelor's degree in any field", "B.S./M.S.") → `graduate`; B.E./B.Tech or "Engineering graduate" → `b.tech`. Degrees without an enum value (B.Pharm, BA, NID, a generic master's, CA-Inter) → `graduate` when it is a bachelor's-level degree, otherwise `null` (noted). |

## Selection

Pools are sampled with a fixed seed per source (`configs/real_data.yaml`). For the annotation budget,
the selection takes per source the first postings in id order with at most 2,400 characters, plus a set
of longer postings (`real-joa-*` and `real-ats-*` above 2,400 characters; the employer-direct source
truncates descriptions at 3,000 characters). Near-duplicates inside the selection (the same boilerplate
with a different title) are rejected so that one employer template cannot dominate.

Web pages (`real-web-*`, Indian job boards in the ODC-By FineWeb crawl) are selected when they have at most
2,400 characters and are not listing pages by a fixed rule (two or more "Key Skills:" blocks, two or more
"..." truncations, or three or more "apply now"/"view job" links). During annotation, pages that turned out
to be listings, reviews or jobs outside India are rejected with the reason.

## Human verification

```bash
jobx verify-real --verifier "<your name>"
```

shows each AI-assisted label that is not verified yet next to the posting, with its alternatives and
notes. Keys: `a` accept, `e` edit a field (value as JSON, e.g. `["python","sql"]` or `null`), `r` reject
with a reason, `v` view the full text, `q` quit. The annotation file is saved after every posting and
`test_real.jsonl` / `real_meta.json` are rebuilt at the end. Re-run `jobx eval` and `jobx results-table`
afterwards; the results table reports how many rows are human-verified.
