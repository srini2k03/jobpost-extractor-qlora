# jobx-qwen2.5-0.5b-lora: job-posting extractor (CPU-trained LoRA)

A LoRA fine-tune of [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). It turns
messy Indian job postings (Naukri/LinkedIn/WhatsApp style, LPA / lakh / per-month salaries, Hinglish fragments)
into one strict JSON object:
`company, role, seniority, skills[], min/max_experience_years, salary{min,max,currency,period}, locations[],
work_mode, employment_type, education`. Fields that are not stated are `null` or `[]`.

The model was **trained, merged, quantised and evaluated entirely on free GitHub-hosted CPU runners**:
- no GPU and no accounts beyond GitHub;
- code: https://github.com/srini2k03/jobpost-extractor-qlora;
- live demo, running in your browser: https://srini2k03.github.io/jobpost-extractor-qlora/.

## Files in this release

| file | what |
|---|---|
| `jobx-qwen2.5-0.5b-lora-adapter.tar.gz` | PEFT LoRA adapter (r=16, alpha=32, all linear projections) |
| `jobx-qwen2.5-0.5b-merged.tar.gz` | adapter merged into the base, fp16 safetensors + tokenizer |
| `jobx-qwen2.5-0.5b-F16.gguf`, `-Q8_0.gguf`, `-Q4_K_M.gguf` | llama.cpp GGUF builds (llama.cpp `b6000`) |

## Prompt

The fine-tune uses a short system prompt. It learned the labelling conventions from the data, so it doesn't
need the long rule list the baselines are given:

```text
Extract the Indian job posting into one JSON object with keys company, role, seniority, skills, min_experience_years, max_experience_years, salary, locations, work_mode, employment_type, education. Use null or [] for anything not stated.
```

The user message is the posting wrapped as in `src/jobx/prompts.py` (`<posting>...</posting>`). Use greedy
decoding (temperature 0). To guarantee valid output, add the JSON-schema grammar
(`schema/jobposting.schema.json`).

## Training

- **Data:** 2177 synthetic postings with labels known by construction, split leakage-safe by template
  family and company (`DATA_CARD.md`). Postings longer than 512 tokens were dropped, not truncated:
  0 of them.
- **Recipe:** LoRA r=16/alpha=32, dropout 0.05, lr 0.0003 with a cosine
  schedule, effective batch 16, completion-only loss (TRL SFTTrainer), bf16 on CPU.
- **Run:** 2 epoch(s), 273 optimiser steps. Measured 34.2 s/step on
  `AMD EPYC 9V45 96-Core Processor` (4 threads). Plan: 2 epoch(s) over all 2177 training examples fit the budget.
- **Curves:** `results/training/qwen2.5-0.5b-lora-cpu/` (loss, learning rate, validation JSON-F1).

## Evaluation

_pending: no measured result yet_

## CPU speed / size / accuracy (llama.cpp, GitHub-hosted runner)

_pending: the evaluation workflow has not run yet_

Metric definitions are in the repository README:
- field exact match after normalisation;
- list P/R/F1;
- hallucination rate = non-null predictions where the gold value is null;
- bootstrap 95% CIs.

## Intended use and limitations

- Structuring job postings for search or analytics, with a human in the loop. It is not for decisions about
  people.
- It was trained on synthetic postings only. Real postings are longer and messier, and the real-postings test
  set is still pending human verification.
- It is a 0.5B model: it is fast and small enough for a browser, but it is less accurate than larger models
  (see the results table in the README).
- Skills and cities follow fixed canonical vocabularies. Salary is compared as INR per annum.

## Licence

The base model is Apache-2.0 (Qwen2.5-0.5B-Instruct). The adapter, the merged weights and the code are under
the same terms; the synthetic data is MIT.
