#### Real postings (`test_real`, headline)

_The real test set is pending: it needs the fetch + draft-label notebook and a human pass with `jobx verify` (see the Colab checklist below)._

| Approach | Hardware | n | JSON valid % | Schema valid % | Field EM % | Micro F1 % [95% CI] | Macro F1 % | Hallucination % [95% CI] | Status |
|---|---|---|---|---|---|---|---|---|---|
| Regex baseline (rules + alias tables) | CPU | - | - | - | - | - | - | - | pending (needs this split) |
| Qwen2.5-1.5B-Instruct zero-shot | T4 | - | - | - | - | - | - | - | pending GPU run |
| Qwen2.5-1.5B-Instruct zero-shot + guided JSON | T4 | - | - | - | - | - | - | - | pending GPU run |
| Qwen2.5-1.5B-Instruct 3-shot | T4 | - | - | - | - | - | - | - | pending GPU run |
| Qwen2.5-1.5B-Instruct 3-shot + guided JSON | T4 | - | - | - | - | - | - | - | pending GPU run |
| Llama-3.2-1B-Instruct 3-shot (optional) | T4 | - | - | - | - | - | - | - | pending GPU run |
| Llama-3.2-1B-Instruct 3-shot + guided JSON (optional) | T4 | - | - | - | - | - | - | - | pending GPU run |
| Qwen2.5-7B-Instruct 3-shot, 4-bit (open-model upper bound) | T4 | - | - | - | - | - | - | - | pending GPU run |
| Qwen2.5-7B-Instruct 3-shot, 4-bit + guided JSON (open-model upper bound) | T4 | - | - | - | - | - | - | - | pending GPU run |
| GPT-OSS-120B via Groq (API, few-shot) | Groq API (free tier) | - | - | - | - | - | - | - | pending API run |
| Qwen2.5-1.5B QLoRA fine-tune (ours) | T4 | - | - | - | - | - | - | - | pending GPU run |
| Qwen2.5-1.5B QLoRA fine-tune + guided JSON | T4 | - | - | - | - | - | - | - | pending GPU run |
| Llama-3.2-1B QLoRA fine-tune (optional) | T4 | - | - | - | - | - | - | - | pending GPU run |
| Fine-tune merged, bitsandbytes 8-bit | T4 | - | - | - | - | - | - | - | pending GPU run |
| Fine-tune merged, AWQ 4-bit | T4 | - | - | - | - | - | - | - | pending GPU run |
| Fine-tune merged, GGUF Q8_0 (llama.cpp) | CPU (GitHub Actions) | - | - | - | - | - | - | - | pending GPU run |
| Fine-tune merged, GGUF Q4_K_M (llama.cpp) | CPU (GitHub Actions) | - | - | - | - | - | - | - | pending GPU run |

#### Synthetic held-out families and companies (`test_synth`)

| Approach | Hardware | n | JSON valid % | Schema valid % | Field EM % | Micro F1 % [95% CI] | Macro F1 % | Hallucination % [95% CI] | Status |
|---|---|---|---|---|---|---|---|---|---|
| Regex baseline (rules + alias tables) | CPU | 471 | 100.0 | 100.0 | 91.1 | 93.1 [92.5, 93.8] | 90.8 | 2.5 [1.7, 3.2] | measured 2026-11-03 @902cc8a |
| Qwen2.5-1.5B-Instruct zero-shot | T4 | - | - | - | - | - | - | - | pending GPU run |
| Qwen2.5-1.5B-Instruct zero-shot + guided JSON | T4 | - | - | - | - | - | - | - | pending GPU run |
| Qwen2.5-1.5B-Instruct 3-shot | T4 | - | - | - | - | - | - | - | pending GPU run |
| Qwen2.5-1.5B-Instruct 3-shot + guided JSON | T4 | - | - | - | - | - | - | - | pending GPU run |
| Llama-3.2-1B-Instruct 3-shot (optional) | T4 | - | - | - | - | - | - | - | pending GPU run |
| Llama-3.2-1B-Instruct 3-shot + guided JSON (optional) | T4 | - | - | - | - | - | - | - | pending GPU run |
| Qwen2.5-7B-Instruct 3-shot, 4-bit (open-model upper bound) | T4 | - | - | - | - | - | - | - | pending GPU run |
| Qwen2.5-7B-Instruct 3-shot, 4-bit + guided JSON (open-model upper bound) | T4 | - | - | - | - | - | - | - | pending GPU run |
| GPT-OSS-120B via Groq (API, few-shot) | Groq API (free tier) | - | - | - | - | - | - | - | incomplete (25/200) |
| Qwen2.5-1.5B QLoRA fine-tune (ours) | T4 | - | - | - | - | - | - | - | pending GPU run |
| Qwen2.5-1.5B QLoRA fine-tune + guided JSON | T4 | - | - | - | - | - | - | - | pending GPU run |
| Llama-3.2-1B QLoRA fine-tune (optional) | T4 | - | - | - | - | - | - | - | pending GPU run |
| Fine-tune merged, bitsandbytes 8-bit | T4 | - | - | - | - | - | - | - | pending GPU run |
| Fine-tune merged, AWQ 4-bit | T4 | - | - | - | - | - | - | - | pending GPU run |
| Fine-tune merged, GGUF Q8_0 (llama.cpp) | CPU (GitHub Actions) | - | - | - | - | - | - | - | pending GPU run |
| Fine-tune merged, GGUF Q4_K_M (llama.cpp) | CPU (GitHub Actions) | - | - | - | - | - | - | - | pending GPU run |

#### Per-field exact match % (`test_synth`)

| Approach | company | role | seniority | skills | min_experience_years | max_experience_years | salary | locations | work_mode | employment_type | education |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Regex baseline (rules + alias tables) | 48.2 | 75.4 | 79.2 | 99.4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

#### Incomplete runs (not part of the comparison above)

Partial runs, or runs with more than 5% failed requests, are scored on what completed and shown here only.

| Approach | Split | Status | Error rate % | n scored | Micro F1 % | JSON valid % |
|---|---|---|---|---|---|---|
| GPT-OSS-120B via Groq (API, few-shot) | test_synth | incomplete (25/200) | 0.0 | 25 | 97.5 | 100.0 |

#### API usage

| Approach | Split | Model | Requests | Input tokens | Output tokens | Cost (USD) |
|---|---|---|---|---|---|---|
| GPT-OSS-120B via Groq (API, few-shot) | test_synth | `openai/gpt-oss-120b` (25) | 25 | 44815 | 11546 | 0.00 |

#### Size / speed / accuracy trade-off

| Variant | Hardware | Size (GB) | Micro F1 % | Tokens/s | Peak memory (GB) | Status |
|---|---|---|---|---|---|---|
| fp16 (transformers) | T4 | - | - | - | - | pending GPU run |
| merged 8-bit (bitsandbytes) | T4 | - | - | - | - | pending GPU run |
| AWQ 4-bit | T4 | - | - | - | - | pending GPU run |
| GGUF Q8_0 (llama.cpp) | CPU, GitHub Actions | - | - | - | - | pending (needs merged model) |
| GGUF Q4_K_M (llama.cpp) | CPU, GitHub Actions | - | - | - | - | pending (needs merged model) |
