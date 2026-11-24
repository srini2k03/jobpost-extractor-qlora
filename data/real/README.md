# Real-postings test set (`test_real`)

Real Indian job postings from licence-checked public sources, held out from everything (no training,
no tuning, no prompt or threshold selection uses them). Built by `jobx real-build`.

| File | What |
|---|---|
| `pool_hf.jsonl` | Postings fetched by `python -m jobx.data.real_fetch` (revision-pinned Hub datasets; licence re-read at fetch time; India by the source's own country/location field) |
| `pool_web.jsonl` | Pages on Indian job boards from an ODC-By web crawl, if the FineWeb probe produced a pool |
| `annotations.jsonl` | One label (or rejection) per selected posting, with acceptable alternatives and notes |
| `test_real.jsonl` | The evaluation split: text + label + alternatives + source + annotation status |
| `rejected.json` | Pool postings left out, with the reason |
| `real_meta.json` | Counts per source, licences, revisions, how many labels are human-verified |

**Labels are AI-assisted annotation, not human-verified** (see `docs/annotation.md` for the rules and
`jobx verify-real` for the human pass). Sources and licences: `real_meta.json`; source probes and the
reasons for including or excluding each candidate dataset: `results/real/` and `docs/real_dataset.md`.
