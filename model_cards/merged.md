---
license: apache-2.0
base_model: {{base_model}}
library_name: transformers
pipeline_tag: text-generation
language: [en, hi]
tags: [qlora, lora, information-extraction, json, job-postings, india]
datasets: [{{artifacts_repo}}]
---

# {{repo_name}}: merged fp16 model for job-posting extraction

The QLoRA adapter `{{adapter_repo}}` merged into `{{base_model}}` (fp16) that turns messy Indian job postings (Naukri/LinkedIn/WhatsApp
style, LPA/lakh/per-month salaries, Hinglish fragments) into one strict JSON object:
`company, role, seniority, skills[], min/max_experience_years, salary{min,max,currency,period}, locations[],
work_mode, employment_type, education`. Fields not stated in the posting are `null`.

Code, data pipeline and evaluation: https://github.com/srini2k03/jobpost-extractor-qlora

## Intended use
- Structuring job postings for search, analytics or recruiting tools, with a human in the loop.
- Not for automated decisions about people. The model reads postings, not candidates.

## Training
- Data: ~{{n_train}} synthetic postings with labels known by construction (plus validated paraphrases if
  present); the split is leakage-safe by template family and company. See `DATA_CARD.md` in the repo.
- Recipe: 4-bit NF4 + double quantisation, fp16 compute (T4), LoRA r={{lora_r}}, alpha={{lora_alpha}},
  dropout 0.05 on all linear projections, lr {{learning_rate}} cosine with 3% warmup, effective batch 16,
  completion-only loss, early stopping on validation JSON micro F1. Details: `docs/hyperparameters.md`.
- Training run: {{wandb_url}}

## Evaluation
{{metrics_table}}

Metric definitions are in the repo README (field EM after normalisation, list P/R/F1, hallucination rate =
non-null predictions where gold is null, bootstrap 95% CIs).

## Limitations
- Trained mostly on synthetic text: long real job descriptions and unusual formats are harder.
- Skills and cities are canonicalised to a fixed vocabulary; skills outside it are returned lower-cased as-is.
- Salary conversion to INR per annum assumes 12 months/year, 2080 hours/year and 83 INR/USD.
- Seniority is only filled when stated in words.

## Licence
The adapter follows the base model's licence ({{base_licence}}). Synthetic training data: MIT.
