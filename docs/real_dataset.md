# Real postings for the headline test set

## Choice

The synthetic test split shows whether the model generalises to unseen *templates and companies*. It cannot
show whether the model works on *real* text, so the headline evaluation is ~200 real public postings.

The fetch script [`src/jobx/data/fetch_real.py`](../src/jobx/data/fetch_real.py) (`jobx fetch-real`, run from
`notebooks/01_paraphrase_7b.ipynb` because the Hugging Face Hub was not reachable from the environment
this repo was built in) tries the candidates in [`configs/real_data.yaml`](../configs/real_data.yaml) in
order:

1. `datastax/linkedin_job_listings`: LinkedIn postings with a free-text `description` and a `location`
   column, so Indian postings can be filtered by location.
2. `jacob-hugging-face/job-descriptions`: job descriptions with company and title columns.

For each candidate the script reads the licence declared on the dataset card and **skips the dataset
unless the licence is in the allowlist** (MIT, Apache-2.0, CC0, CC-BY(-SA) 4.0, ODC-By, PDDL, CDLA-Permissive,
ODbL). It then keeps postings that look Indian (an Indian city or "India" in the location, or INR/LPA/lakh
language in the text) with 200–3,000 characters, removes exact duplicates and samples 200 with a fixed seed.

## TODO before committing any real text

- [ ] **Licence check (not done; could not be verified from the build environment).** Open the chosen
      dataset's card, confirm the declared licence, and check that the underlying source (e.g. LinkedIn
      listings) allows redistribution of the text. If in doubt, keep `data/real/test_real.jsonl` out of the
      public repo, keep the artefacts repo private, and publish only metrics.
- [ ] Record the dataset id, revision and licence here.
- [ ] Verify the drafted labels: `jobx pull`, then `jobx verify` (keys: a accept, e edit, s skip, v view, q quit).
      Progress is saved after every posting.

`data/real/raw_postings.jsonl` is git-ignored on purpose.

## Labelling protocol

Drafts come from Qwen2.5-7B-Instruct (4-bit) prompted 3-shot with the project prompt. The human verifier
applies the conventions in the system prompt ([`src/jobx/prompts.py`](../src/jobx/prompts.py)):

- `null` unless stated;
- role without seniority words;
- skills named anywhere, including the title;
- salary as absolute amounts in the stated period;
- the job's cities, not the head office.

Postings that are not job ads, or not Indian, are skipped (`s`). The verifier name and edited fields are
stored per row.
