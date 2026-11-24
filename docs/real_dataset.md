# Real postings: how the test set was sourced

**Status:** 130 real Indian postings in `data/real/test_real.jsonl`, labelled with AI assistance and **not
human-verified** (`jobx verify-real` is the human pass). Annotation rules and record format:
[`docs/annotation.md`](annotation.md).

## How sources were chosen (licence first)

1. `real_probe.yml` (job `datasets`) searched the Hugging Face Hub for job-posting datasets and read each
   dataset card's licence and gating flag **before** reading any rows. Only datasets whose licence is in the
   allowlist (CC0, CC-BY(-SA) 4.0, ODC-By, MIT, Apache-2.0, PDDL, CDLA-Permissive, ODbL) and that are not gated
   were streamed. Report: [`results/real/probe_datasets.json`](../results/real/probe_datasets.json) (173 datasets).
2. Job `details` read the cards (provenance) of a shortlist and counted Indian rows by each dataset's own
   country/location field: [`results/real/probe_details.json`](../results/real/probe_details.json).
3. Job `web` scanned only the `url` column of FineWeb `sample/10BT` (ODC-By; 14.6 M documents) for pages on
   Indian job boards and read the text of a capped sample of 450 hits:
   [`results/real/probe_web.json`](../results/real/probe_web.json), hits in `results/real/web_hits.jsonl`.

## Included

| Source | Licence | Why |
|---|---|---|
| `victor/real-or-fake-fake-jobposting-prediction` (EMSCAD, University of the Aegean) | CC0-1.0 | real employer ads (2012-14); `fraudulent == 0` only; India by `location` starting `IN,` |
| `JobOpportunitiesAPI/joa-employer-direct-job-postings-2026-10` | CC-BY-4.0 | postings from employers' own career pages and ATS feeds; `country == IN`; descriptions truncated at 3,000 characters by the source |
| `makenomistakesllc/tech-job-postings-9-ats` | CC-BY-4.0 | public company ATS boards; `country == IN` |
| `HuggingFaceFW/fineweb` (sample/10BT) | ODC-By (Common Crawl terms apply) | single job ads on Indian job boards (fresherslive, timesjobs, monsterindia, placementindia, ...) |

Revisions are pinned in [`configs/real_data.yaml`](../configs/real_data.yaml); every row keeps its source
row id (or URL and crawl), and `real_meta.json` lists counts and licences for attribution.

## Excluded on purpose

- `logiover/naukri-job-scraper-sample-data` (CC-BY on its card): its card says the data was scraped through
  residential proxies to bypass anti-bot protection, i.e. against the site's terms.
- `NextGig-Rocks/global-job-postings-multi-ats` (CC-BY-4.0): the descriptions were rewritten by an LLM, so they
  are not raw postings.
- LinkedIn scrapes and datasets without a licence on the card.

## Selection

Hub postings: a fixed-seed sample per source, then for the annotation budget the first postings in id order
of at most 2,400 characters plus a set of longer ones. Web pages: single ads of at most 2,400 characters that
are not listing pages by a fixed rule. 14 selected postings were rejected during annotation (listing pages, a
review page, one job in Dubai, near-duplicate ATS boilerplate); see `data/real/rejected.json`.
