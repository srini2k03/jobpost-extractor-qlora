---
license: apache-2.0
base_model: {{merged_repo}}
pipeline_tag: text-generation
tags: [awq, information-extraction, json, job-postings, india]
---

# {{repo_name}}: AWQ 4-bit build

AutoAWQ 4-bit (GEMM, group size 128) quantisation of `{{merged_repo}}`, calibrated on training postings only. Load with `AutoAWQForCausalLM.from_quantized` or vLLM.

## T4 benchmark
{{benchmark_table}}

## Evaluation
{{metrics_table}}

## Limitations and licence
As for the merged model: see https://github.com/srini2k03/jobpost-extractor-qlora. Licence follows the base
model ({{base_licence}}).
