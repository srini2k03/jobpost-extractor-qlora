---
title: Job Posting to JSON
emoji: 🧾
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: Messy Indian job postings to strict JSON (QLoRA, GGUF, CPU)
---

# Job posting to JSON

Demo for [jobpost-extractor-qlora](https://github.com/srini2k03/jobpost-extractor-qlora): a QLoRA fine-tune
of Qwen2.5-1.5B-Instruct served as GGUF Q4_K_M with llama.cpp on the free CPU tier.

This folder is deployed by `.github/workflows/deploy_space.yml`, which copies `src/jobx` next to `app.py`.
Set the Space variable `JOBX_GGUF_REPO` to `<your-hf-username>/jobx-qwen2.5-1.5b-gguf`. Until the GGUF
exists, the app falls back to the rule-based extractor and says so on the page.
