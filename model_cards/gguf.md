---
license: apache-2.0
base_model: {{merged_repo}}
pipeline_tag: text-generation
tags: [gguf, llama.cpp, information-extraction, json, job-postings, india]
---

# {{repo_name}}: GGUF builds for CPU inference

GGUF conversions of `{{merged_repo}}` (QLoRA fine-tune of `{{base_model}}`) made with llama.cpp
`{{llama_cpp_ref}}` in GitHub Actions:

| file | quantisation | size |
|---|---|---|
| `jobx-qwen2.5-1.5b-Q8_0.gguf` | Q8_0 | {{size_q8}} |
| `jobx-qwen2.5-1.5b-Q4_K_M.gguf` | Q4_K_M | {{size_q4}} |

## Use
```bash
ollama create jobx -f ollama/Modelfile          # Modelfile in the GitHub repo
llama-server -m jobx-qwen2.5-1.5b-Q4_K_M.gguf   # OpenAI-compatible server
```
The system prompt must match `src/jobx/prompts.py` (the Modelfile already contains it).

## CPU benchmark (GitHub Actions runner)
{{benchmark_table}}

## Evaluation
{{metrics_table}}

## Limitations and licence
As for the merged model: see https://github.com/srini2k03/jobpost-extractor-qlora. Licence follows the base
model ({{base_licence}}).
