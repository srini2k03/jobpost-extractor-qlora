#### Real postings (`test_real`, headline)

_130 real Indian job postings from 4 licence-checked public sources (cc-by-4.0, cc0-1.0, odc-by); see `data/real/real_meta.json`. Labels: AI-assisted annotation, not human-verified (0 of 130 human-verified); 100 rows list acceptable alternatives for ambiguous fields (`docs/annotation.md`)._

| Approach | Hardware | n | JSON valid % | Schema valid % | Field EM % | Micro F1 % [95% CI] | Macro F1 % | Hallucination % [95% CI] | Status |
|---|---|---|---|---|---|---|---|---|---|
| Regex baseline (rules + alias tables) | CPU | 130 | 100.0 | 100.0 | 75.5 | 60.4 [58.0, 63.1] | 74.1 | 5.5 [3.6, 7.7] | measured 2026-11-24 @52f5f88 |
| Qwen2.5-0.5B-Instruct zero-shot | CPU (GitHub Actions) | 130 | 86.2 | 0.0 | 39.7 | 32.0 [29.3, 34.3] | 32.4 | 58.4 [53.5, 62.6] | measured 2026-11-24 @c66bc22 |
| Qwen2.5-0.5B-Instruct zero-shot + guided JSON | CPU (GitHub Actions) | - | - | - | - | - | - | - | pending CPU run |
| Qwen2.5-0.5B-Instruct 3-shot | CPU (GitHub Actions) | - | - | - | - | - | - | - | pending CPU run |
| Qwen2.5-0.5B-Instruct 3-shot + guided JSON | CPU (GitHub Actions) | 130 | 70.0 | 67.7 | 53.0 | 35.7 [32.2, 39.4] | 37.3 | 23.9 [20.3, 27.4] | measured 2026-11-24 @c66bc22 |
| Qwen2.5-1.5B-Instruct zero-shot | CPU (GitHub Actions) | 130 | 93.8 | 13.8 | 62.3 | 50.2 [47.2, 53.2] | 58.5 | 35.4 [31.3, 39.4] | measured 2026-11-24 @c66bc22 |
| Qwen2.5-1.5B-Instruct 3-shot | CPU (GitHub Actions) | - | - | - | - | - | - | - | pending CPU run |
| Qwen2.5-0.5B LoRA fine-tune (ours), GGUF F16 | CPU (GitHub Actions) | 130 | 96.9 | 86.2 | 75.2 | 56.1 [52.8, 59.2] | 71.7 | 7.9 [5.6, 10.4] | measured 2026-11-24 @c66bc22 |
| Qwen2.5-0.5B LoRA fine-tune (ours), GGUF Q8_0 | CPU (GitHub Actions) | - | - | - | - | - | - | - | pending CPU run |
| Qwen2.5-0.5B LoRA fine-tune (ours), GGUF Q4_K_M (browser demo) | CPU (GitHub Actions) | 130 | 94.6 | 85.4 | 74.7 | 54.4 [51.0, 57.9] | 71.0 | 7.6 [5.5, 9.8] | measured 2026-11-24 @c66bc22 |
| Qwen2.5-0.5B LoRA fine-tune (ours), GGUF Q4_K_M + JSON grammar (browser setting) | CPU (GitHub Actions) | 130 | 96.2 | 92.3 | 75.3 | 56.0 [52.8, 59.1] | 72.0 | 7.6 [5.5, 9.8] | measured 2026-11-24 @c66bc22 |
| Qwen2.5-3B-Instruct zero-shot (GGUF Q4_K_M) | CPU (GitHub Actions) | 130 | 98.5 | 59.2 | 70.2 | 56.4 [53.2, 59.3] | 67.1 | 30.3 [26.9, 33.7] | measured 2026-11-24 @c66bc22 |
| GPT-OSS-120B via Groq (API, few-shot) | Groq API (free tier) | - | - | - | - | - | - | - | incomplete (9/130) |

_15 optional GPU-route rows (1.5B/7B baselines, 1.5B QLoRA, AWQ; need a GPU plus Hugging Face and W&B accounts) were not run; see `docs/gpu_route.md`._

#### Synthetic held-out families and companies (`test_synth`)

| Approach | Hardware | n | JSON valid % | Schema valid % | Field EM % | Micro F1 % [95% CI] | Macro F1 % | Hallucination % [95% CI] | Status |
|---|---|---|---|---|---|---|---|---|---|
| Regex baseline (rules + alias tables) | CPU | 471 | 100.0 | 100.0 | 91.1 | 93.1 [92.5, 93.8] | 90.8 | 2.5 [1.7, 3.2] | measured 2026-11-03 @902cc8a |
| Qwen2.5-0.5B-Instruct zero-shot | CPU (GitHub Actions) | 471 | 96.2 | 0.0 | 54.2 | 71.6 [70.4, 72.7] | 56.2 | 67.1 [64.2, 69.8] | measured 2026-11-03 @971e1dc |
| Qwen2.5-0.5B-Instruct zero-shot + guided JSON | CPU (GitHub Actions) | 200 | 99.5 | 77.5 | 54.7 | 72.1 [70.6, 73.6] | 54.5 | 73.0 [69.4, 76.6] | measured 2026-11-03 @971e1dc |
| Qwen2.5-0.5B-Instruct 3-shot | CPU (GitHub Actions) | 471 | 99.4 | 23.6 | 66.0 | 79.1 [78.1, 80.0] | 65.1 | 43.7 [41.4, 46.4] | measured 2026-11-03 @971e1dc |
| Qwen2.5-0.5B-Instruct 3-shot + guided JSON | CPU (GitHub Actions) | 200 | 99.0 | 99.0 | 66.5 | 80.1 [78.5, 81.4] | 66.2 | 44.2 [40.6, 48.1] | measured 2026-11-03 @971e1dc |
| Qwen2.5-1.5B-Instruct zero-shot | CPU (GitHub Actions) | 471 | 100.0 | 5.1 | 84.4 | 91.4 [90.8, 92.1] | 86.0 | 35.3 [32.6, 37.6] | measured 2026-11-03 @971e1dc |
| Qwen2.5-1.5B-Instruct 3-shot | CPU (GitHub Actions) | 471 | 97.5 | 40.3 | 84.0 | 89.8 [88.8, 90.8] | 85.8 | 21.7 [19.8, 23.7] | measured 2026-11-03 @971e1dc |
| Qwen2.5-0.5B LoRA fine-tune (ours), GGUF F16 | CPU (GitHub Actions) | 471 | 100.0 | 99.6 | 98.8 | 99.3 [99.1, 99.5] | 99.0 | 1.9 [1.2, 2.6] | measured 2026-11-04 @fcdc48b |
| Qwen2.5-0.5B LoRA fine-tune (ours), GGUF Q8_0 | CPU (GitHub Actions) | 471 | 100.0 | 99.8 | 98.8 | 99.3 [99.1, 99.5] | 98.9 | 1.7 [1.0, 2.3] | measured 2026-11-04 @fcdc48b |
| Qwen2.5-0.5B LoRA fine-tune (ours), GGUF Q4_K_M (browser demo) | CPU (GitHub Actions) | 471 | 100.0 | 99.8 | 98.5 | 99.0 [98.8, 99.2] | 98.5 | 2.0 [1.3, 2.7] | measured 2026-11-24 @c66bc22 |
| Qwen2.5-0.5B LoRA fine-tune (ours), GGUF Q4_K_M + JSON grammar (browser setting) | CPU (GitHub Actions) | - | - | - | - | - | - | - | pending CPU run |
| Qwen2.5-3B-Instruct zero-shot (GGUF Q4_K_M) | CPU (GitHub Actions) | - | - | - | - | - | - | - | pending CPU run |
| GPT-OSS-120B via Groq (API, few-shot) | Groq API (free tier) | 200 | 100.0 | 100.0 | 96.9 | 97.9 [97.4, 98.4] | 97.7 | 1.0 [0.3, 1.8] | measured 2026-11-06 @dd40a42 |

_15 optional GPU-route rows (1.5B/7B baselines, 1.5B QLoRA, AWQ; need a GPU plus Hugging Face and W&B accounts) were not run; see `docs/gpu_route.md`._

#### Per-field exact match % (`test_real`)

| Approach | company | role | seniority | skills | min_experience_years | max_experience_years | salary | locations | work_mode | employment_type | education |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Regex baseline (rules + alias tables) | 45.4 | 50.0 | 95.4 | 5.4 | 86.2 | 88.5 | 96.9 | 92.3 | 97.7 | 93.8 | 79.2 |
| Qwen2.5-0.5B-Instruct zero-shot | 74.6 | 77.7 | 26.2 | 0.8 | 52.3 | 70.0 | 49.2 | 19.2 | 16.9 | 37.7 | 12.3 |
| Qwen2.5-0.5B-Instruct 3-shot + guided JSON | 42.3 | 63.1 | 73.1 | 2.3 | 67.7 | 80.8 | 84.6 | 27.7 | 35.4 | 46.2 | 60.0 |
| Qwen2.5-1.5B-Instruct zero-shot | 70.0 | 82.3 | 80.0 | 6.2 | 76.2 | 95.4 | 92.3 | 50.0 | 26.9 | 53.8 | 52.3 |
| Qwen2.5-0.5B LoRA fine-tune (ours), GGUF F16 | 80.8 | 79.2 | 85.4 | 6.9 | 93.1 | 92.3 | 90.0 | 30.8 | 96.9 | 95.4 | 76.9 |
| Qwen2.5-0.5B LoRA fine-tune (ours), GGUF Q4_K_M (browser demo) | 82.3 | 80.0 | 85.4 | 6.9 | 88.5 | 90.0 | 89.2 | 30.0 | 97.7 | 93.8 | 77.7 |
| Qwen2.5-0.5B LoRA fine-tune (ours), GGUF Q4_K_M + JSON grammar (browser setting) | 83.8 | 81.5 | 85.4 | 6.9 | 90.8 | 90.0 | 89.2 | 30.0 | 97.7 | 94.6 | 78.5 |
| Qwen2.5-3B-Instruct zero-shot (GGUF Q4_K_M) | 80.0 | 86.2 | 83.8 | 6.2 | 95.4 | 85.4 | 90.8 | 76.9 | 51.5 | 43.8 | 72.3 |

#### Per-field exact match % (`test_synth`)

| Approach | company | role | seniority | skills | min_experience_years | max_experience_years | salary | locations | work_mode | employment_type | education |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Regex baseline (rules + alias tables) | 48.2 | 75.4 | 79.2 | 99.4 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Qwen2.5-0.5B-Instruct zero-shot | 84.7 | 93.6 | 42.3 | 77.1 | 25.7 | 29.3 | 52.4 | 67.7 | 57.5 | 49.0 | 17.0 |
| Qwen2.5-0.5B-Instruct zero-shot + guided JSON | 88.0 | 96.0 | 44.5 | 90.0 | 21.0 | 25.0 | 54.5 | 70.0 | 56.5 | 49.5 | 6.5 |
| Qwen2.5-0.5B-Instruct 3-shot | 55.6 | 97.7 | 51.2 | 74.5 | 92.4 | 86.2 | 52.0 | 70.1 | 49.5 | 44.6 | 51.8 |
| Qwen2.5-0.5B-Instruct 3-shot + guided JSON | 61.0 | 96.0 | 46.5 | 74.5 | 89.0 | 82.5 | 51.5 | 70.0 | 60.0 | 53.0 | 47.0 |
| Qwen2.5-1.5B-Instruct zero-shot | 83.4 | 99.2 | 78.6 | 98.1 | 93.6 | 89.2 | 80.5 | 92.6 | 66.7 | 62.2 | 84.1 |
| Qwen2.5-1.5B-Instruct 3-shot | 86.6 | 97.2 | 74.7 | 81.1 | 93.2 | 87.9 | 76.4 | 89.2 | 77.1 | 68.8 | 91.9 |
| Qwen2.5-0.5B LoRA fine-tune (ours), GGUF F16 | 96.8 | 97.9 | 99.6 | 99.4 | 99.6 | 99.8 | 98.5 | 99.4 | 96.2 | 99.8 | 100.0 |
| Qwen2.5-0.5B LoRA fine-tune (ours), GGUF Q8_0 | 96.8 | 97.9 | 99.6 | 99.4 | 99.8 | 100.0 | 98.1 | 99.4 | 96.4 | 99.8 | 100.0 |
| Qwen2.5-0.5B LoRA fine-tune (ours), GGUF Q4_K_M (browser demo) | 96.6 | 98.1 | 99.4 | 99.4 | 99.8 | 100.0 | 95.8 | 99.4 | 95.5 | 99.8 | 100.0 |
| GPT-OSS-120B via Groq (API, few-shot) | 99.0 | 97.5 | 90.0 | 88.5 | 100.0 | 100.0 | 98.0 | 98.5 | 99.5 | 100.0 | 95.0 |

#### Incomplete runs (not part of the comparison above)

Partial runs, or runs with more than 5% failed requests, are scored on what completed and shown here only.

| Approach | Split | Status | Error rate % | n scored | Micro F1 % | JSON valid % |
|---|---|---|---|---|---|---|
| GPT-OSS-120B via Groq (API, few-shot) | test_real | incomplete (9/130) | 0.0 | 9 | 74.0 | 100.0 |

#### API usage

| Approach | Split | Model | Requests | Input tokens | Output tokens | Cost (USD) |
|---|---|---|---|---|---|---|
| GPT-OSS-120B via Groq (API, few-shot) | test_real | `openai/gpt-oss-120b` (9) | 9 | 19112 | 4650 | 0.00 |
| GPT-OSS-120B via Groq (API, few-shot) | test_synth | `openai/gpt-oss-120b` (200) | 200 | 368572 | 99223 | 0.00 |

#### Size / speed / accuracy trade-off

| Variant | Hardware | Size (GB) | Micro F1 % | Tokens/s | Peak memory (GB) | Status |
|---|---|---|---|---|---|---|
| LoRA 0.5B GGUF F16 (llama.cpp) | CPU (INTEL(R) XEON(R) PLATINUM 8573C, 4 threads, GitHub Actions) | 0.994 | 99.3 | 25.06 | 1.042 | measured 2026-11-04 |
| LoRA 0.5B GGUF Q8_0 (llama.cpp) | CPU (AMD EPYC 7763 64-Core Processor, 4 threads, GitHub Actions) | 0.531 | 99.3 | 52.19 | 0.611 | measured 2026-11-04 |
| LoRA 0.5B GGUF Q4_K_M (llama.cpp) | CPU (AMD EPYC 7763 64-Core Processor, 4 threads, GitHub Actions) | 0.398 | 99.0 | 56.11 | 0.514 | measured 2026-11-04 |

#### Robustness suite (`robustness`, 14 categories x 12 generated cases)

_Micro-F1 % per category (gold by construction). "Not a job post": share of outputs with every field empty. Injection success: share of prompt-injection cases where an injected value reached the output (lower is better)._

| Category | Regex rules | 0.5B zero-shot | 0.5B 3-shot + grammar | 1.5B zero-shot | 3B zero-shot | LoRA 0.5B F16 | LoRA 0.5B Q4_K_M | LoRA Q4_K_M + grammar |
|---|---|---|---|---|---|---|---|---|
| Missing salary | 89.9 | 71.3 | 84.1 | 99.1 | 91.8 | 100.0 | 100.0 | 100.0 |
| Several salaries | 93.8 | 70.2 | 77.8 | 95.4 | 92.1 | 100.0 | 100.0 | 100.0 |
| Conflicting salary | 90.9 | 58.4 | 75.0 | 95.1 | 91.3 | 100.0 | 100.0 | 100.0 |
| Monthly vs annual | 93.0 | 73.0 | 81.9 | 94.4 | 87.1 | 98.3 | 99.2 | 99.2 |
| Several locations | 96.2 | 74.8 | 81.6 | 99.3 | 92.7 | 100.0 | 100.0 | 100.0 |
| Work-mode phrasing | 94.7 | 69.6 | 84.2 | 95.3 | 91.3 | 100.0 | 99.6 | 99.6 |
| Contradictory experience | 95.0 | 73.1 | 75.8 | 98.7 | 88.1 | 100.0 | 100.0 | 100.0 |
| No job title | 88.9 | 58.0 | 69.9 | 77.3 | 80.8 | 99.1 | 98.6 | 98.6 |
| No company | 93.6 | 72.3 | 81.7 | 87.0 | 85.0 | 100.0 | 100.0 | 100.0 |
| Hindi / English mix | 83.5 | 65.4 | 76.5 | 82.7 | 88.1 | 98.3 | 95.0 | 95.0 |
| Very long (~7k chars) | 74.5 | 69.7 | 67.7 | 81.1 | 89.0 | 97.5 | 96.0 | 96.0 |
| Malformed (HTML, caps, no punctuation) | 85.8 | 67.9 | 75.0 | 94.4 | 92.1 | 99.2 | 100.0 | 100.0 |
| Not a job post | 0.0 | 0.0 | 0.0 | 16.7 | 83.3 | 16.7 | 16.7 | 16.7 |
| Prompt injection | 90.8 | 52.7 | 73.4 | 74.5 | 86.7 | 96.8 | 96.0 | 96.0 |
| **Injection success %** | 16.7 | 66.7 | 33.3 | 66.7 | 41.7 | 33.3 | 41.7 | 41.7 |

#### Field-confidence calibration (`qwen2.5-0.5b-lora-cpu_q4_k_m`)

_Fitted on val (3872 fields); abstention threshold 0.00 = smallest with 95% selective accuracy on val. Held-out splits not used for fitting._

| Split | Fields | ECE raw | ECE calibrated | Brier calibrated | AUROC | Kept at threshold % | Accuracy kept % | Accuracy all % |
|---|---|---|---|---|---|---|---|---|
| val | 3872 | 0.014 | 0.007 | 0.009 | 0.956 | 100.0 | 98.1 | 98.1 |
| test_synth | 5181 | 0.012 | 0.008 | 0.007 | 0.9131 | 100.0 | 98.5 | 98.5 |
| test_real | 1353 | 0.199 | 0.101 | 0.112 | 0.8797 | 96.8 | 78.9 | 76.4 |
| robustness | 1848 | 0.021 | 0.011 | 0.017 | 0.7186 | 100.0 | 97.6 | 97.6 |

#### Quantisation sweep (one runner, all variants)

_CPU (Intel(R) Xeon(R) 6975P-C, 4 threads, GitHub Actions), llama.cpp b6000, measured 2026-11-24 @2239191; accuracy on 150 evenly spaced test_synth and 109 test_real postings._

| GGUF | Size (GB) | Synth micro-F1 % | Real micro-F1 % | JSON valid % (real) | Gen tokens/s | Prompt tokens/s | p50 latency (s) | p95 latency (s) | Cold start (s) | Peak RAM (GB) |
|---|---|---|---|---|---|---|---|---|---|---|
| F16 | 0.994 | 99.6 | 54.9 | 96.3 | 29.56 | 173.5 | 4.818 | 6.895 | 2.065 | 1.069 |
| Q8_0 | 0.531 | 99.6 | 54.9 | 97.2 | 42.3 | 198.0 | 3.513 | 5.434 | 2.031 | 0.637 |
| Q6_K | 0.506 | 99.5 | 55.6 | 97.2 | 42.84 | 165.8 | 3.592 | 5.887 | 2.031 | 0.614 |
| Q5_K_M | 0.42 | 99.5 | 54.9 | 96.3 | 40.65 | 105.6 | 4.141 | 8.124 | 2.033 | 0.534 |
| Q4_K_M | 0.398 | 99.4 | 53.4 | 94.5 | 40.16 | 152.5 | 3.873 | 6.431 | 2.031 | 0.541 |
| Q4_0 | 0.352 | 99.3 | 53.9 | 96.3 | 57.35 | 235.5 | 2.636 | 4.422 | 2.031 | 0.656 |
| Q3_K_M | 0.355 | 99.0 | 54.3 | 95.4 | 43.83 | 183.1 | 3.45 | 5.501 | 2.032 | 0.526 |
| Q2_K | 0.339 | 98.3 | 54.1 | 95.4 | 42.91 | 153.6 | 3.66 | 6.205 | 2.031 | 0.458 |

| Threads (Q4_K_M) | Gen tokens/s | Prompt tokens/s | p50 latency (s) |
|---|---|---|---|
| 1 | 14.92 | 77.9 | 8.79 |
| 2 | 27.68 | 152.8 | 4.652 |
| 4 | 40.03 | 154.9 | 3.477 |

#### LoRA study (reduced budget, validation only)

_Reduced-budget sweep: 640 training examples, 40 steps (effective batch 16) per configuration; the released model used 2,177 examples x 2 epochs (273 steps). Selected: `r16_lr6e-4` (within its 95% CI: none)._

| Config | r / alpha | Targets | LR | Trainable params | % of total | Adapter (MB) | s/step | Peak RAM (GB) | Val loss | Val micro-F1 % [95% CI] | JSON valid % | Hallucination % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r16_lr6e-4 | 16 / 32 | all linear | 0.0006 | 8,798,208 | 1.78 | 35.24 | 47.8 | 4.58 | 0.0248 | 96.5 [95.8, 97.3] | 100.0 | 4.1 |
| r32_all | 32 / 64 | all linear | 0.0003 | 17,596,416 | 3.56 | 70.43 | 50.33 | 4.94 | 0.024 | 95.5 [94.4, 96.5] | 100.0 | 3.2 |
| r16_all | 16 / 32 | all linear | 0.0003 | 8,798,208 | 1.78 | 35.24 | 47.28 | 4.59 | 0.0361 | 93.9 [92.8, 95.0] | 100.0 | 6.2 |
| r8_all | 8 / 16 | all linear | 0.0003 | 4,399,104 | 0.89 | 17.64 | 47.4 | 4.5 | 0.0525 | 92.1 [90.7, 93.3] | 100.0 | 6.6 |
| r16_lr1e-4 | 16 / 32 | all linear | 0.0001 | 8,798,208 | 1.78 | 35.24 | 81.69 | 4.31 | 0.0734 | 88.0 [86.5, 89.5] | 100.0 | 11.2 |
| r16_attn | 16 / 32 | attention | 0.0003 | 2,162,688 | 0.44 | 8.68 | 136.99 | 4.11 | 0.0799 | 87.8 [85.9, 89.5] | 99.3 | 9.2 |
