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
| `jobx-qwen2.5-0.5b-lora-adapter.tar.gz` | PEFT LoRA adapter (r={{lora_r}}, alpha={{lora_alpha}}, all linear projections) |
| `jobx-qwen2.5-0.5b-merged.tar.gz` | adapter merged into the base, fp16 safetensors + tokenizer |
| `jobx-qwen2.5-0.5b-F16.gguf`, `-Q8_0.gguf`, `-Q4_K_M.gguf` | llama.cpp GGUF builds (llama.cpp `{{llama_cpp_ref}}`) |

## Prompt

The fine-tune uses a short system prompt. It learned the labelling conventions from the data, so it doesn't
need the long rule list the baselines are given:

```text
{{system_prompt}}
```

The user message is the posting wrapped as in `src/jobx/prompts.py` (`<posting>...</posting>`). Use greedy
decoding (temperature 0). To guarantee valid output, add the JSON-schema grammar
(`schema/jobposting.schema.json`).

## Training

- **Data:** {{n_train}} synthetic postings with labels known by construction, split leakage-safe by template
  family and company (`DATA_CARD.md`). Postings longer than {{max_length}} tokens were dropped, not truncated:
  {{dropped}} of them.
- **Recipe:** LoRA r={{lora_r}}/alpha={{lora_alpha}}, dropout 0.05, lr {{learning_rate}} with a cosine
  schedule, effective batch 16, completion-only loss (TRL SFTTrainer), {{precision}} on CPU.
- **Run:** {{epochs}} epoch(s), {{max_steps}} optimiser steps. Measured {{seconds_per_step}} s/step on
  `{{cpu}}` ({{threads}} threads). Plan: {{plan_reason}}.
- **Curves:** `results/training/qwen2.5-0.5b-lora-cpu/` (loss, learning rate, validation JSON-F1).

## Evaluation

{{metrics_table}}

## CPU speed / size / accuracy (llama.cpp, GitHub-hosted runner)

{{benchmark_table}}

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
