# Optional GPU route (not run)

The headline results come from the **no-account CPU route**: everything runs in GitHub Actions, with no
accounts beyond GitHub (see the README). This page documents the original GPU route. Its code, notebooks and
configs are complete and tested on CPU, but **it has not been run**, because it needs a GPU (Colab T4 or a
local NVIDIA GPU) plus Hugging Face and Weights & Biases accounts.

What it adds on top of the CPU route:
- a 7B paraphrase pass;
- 1.5B and 7B baselines with lm-format-enforcer guided decoding;
- a QLoRA fine-tune of Qwen2.5-1.5B with W&B logging;
- AWQ and bitsandbytes benchmarks;
- Hub-hosted model repos and a Gradio Space.

## Baselines and upper bound (notebook 02)

Qwen2.5-1.5B zero-shot and 3-shot, an optional Llama-3.2-1B, and Qwen2.5-7B-Instruct 3-shot in 4-bit as the
open-model upper bound. Each runs with free decoding and with guided JSON decoding (lm-format-enforcer).

## QLoRA fine-tune (notebook 03, [`src/jobx/train/sft.py`](../src/jobx/train/sft.py))

The recipe is QLoRA
(4-bit NF4 with double quantisation, fp16 compute because the T4 has no bf16) with TRL `SFTTrainer`.
The loss covers completion tokens only, and LoRA r=16/α=32 is applied to all linear projections, with
gradient checkpointing and paged 8-bit AdamW. The learning rate is 2e-4 with a cosine schedule and 3%
warmup, and the effective batch is 16. Early stopping uses validation JSON micro F1, computed by greedy
decoding at each evaluation. Weights & Biases logging is on. The choices are explained in
[`docs/hyperparameters.md`](hyperparameters.md), and a small r/lr sweep is included.


## Quantisation

The merged model is converted to GGUF Q8_0 and Q4_K_M, with llama.cpp built inside
GitHub Actions ([`quantize_benchmark.yml`](../.github/workflows/quantize_benchmark.yml)), and benchmarked on the
runner's CPU: tokens/s, peak RAM and accuracy. AWQ 4-bit and bitsandbytes 8-bit are benchmarked on the
T4 (notebook 05).


## Serving

- FastAPI `POST /extract` ([`src/jobx/serve/app.py`](../src/jobx/serve/app.py)) uses llama-cpp-python (GGUF,
  CPU), validates against the schema and re-prompts on invalid output. It also offers optional
  grammar-guided decoding. Dockerfile in [`docker/`](../docker/Dockerfile).
- Ollama: [`ollama/Modelfile`](../ollama/Modelfile), generated from the same prompt.
- Gradio Space ([`space/`](../space/)) with examples, a JSON view and field highlighting. It can be deployed with
  `deploy_space.yml`, which needs an `HF_TOKEN` secret (a Hugging Face account).
- Model cards for the adapter, merged, GGUF and AWQ repos: [`model_cards/`](../model_cards/).


## Run on your own GPU

The GPU steps also run on a local NVIDIA GPU. The setup was written for an RTX 4080 Laptop GPU (12 GB, bf16)
under Windows + WSL2, and the full walkthrough is in [`docs/local_gpu_setup.md`](local_gpu_setup.md).

```bash
make local-setup && make local-check     # pinned CUDA 12.8 stack, then a PASS/FAIL environment table
make local-paraphrase local-baselines local-train local-eval local-awq local-push   # or: make local-all
```

The make targets call the same `jobx.pipeline` functions as the notebooks, write to the same paths and push
to the same Hub repos, using the bf16 profile in [`configs/gpu/local_rtx4080.yaml`](../configs/gpu/local_rtx4080.yaml)
and [`configs/train/local_rtx4080.yaml`](../configs/train/local_rtx4080.yaml). Each step can be restarted after
an interruption. Expect about 3–4 hours of GPU time on a 4080 laptop. The Colab notebooks below remain the
alternative.


## Colab checklist (in order)

Secrets in Colab (key icon, with *Notebook access* on): `HF_TOKEN` (write), `HF_USERNAME`, `WANDB_API_KEY`.
Each notebook opens with an *Open in Colab* badge and run-all instructions, and pushes its outputs to repos
under your Hugging Face account (the `jobx-artifacts` dataset repo is private). Afterwards, `jobx pull` downloads them into this repo, and
`jobx eval && jobx results-table` updates the tables.

| # | Notebook | Runtime | Secrets | Produces |
|---|---|---|---|---|
| 1 | `01_paraphrase_7b` | T4, ~1.5 h | HF_TOKEN, HF_USERNAME | validated paraphrases + filter stats; real-postings drafts |
| - | `jobx pull && jobx verify` (your machine) | ~30 min | HF_TOKEN | `data/real/test_real.jsonl` (human-verified) |
| 2 | `02_baselines` | T4, ~2–2.5 h (can be split) | HF_TOKEN, HF_USERNAME | zero/few-shot (+guided) predictions and metrics, 1.5B and 7B |
| 3 | `03_train_qlora` | T4, ~1–1.5 h | HF_TOKEN, HF_USERNAME, WANDB_API_KEY | LoRA adapter on the Hub, W&B run, log history + curves |
| 4 | `04_eval_predict` | T4, ~45–60 min | HF_TOKEN, HF_USERNAME | fine-tune predictions (+guided), merged fp16 model on the Hub |
| 5 | Actions → `quantize-benchmark` | CPU runner, ~1–2 h | repo secret HF_TOKEN, variable HF_USERNAME | GGUF Q8_0/Q4_K_M on the Hub, CPU benchmark + predictions committed |
| 6 | `05_awq_benchmark` | T4, ~45 min | HF_TOKEN, HF_USERNAME | AWQ model, T4 benchmark rows (fp16 / 8-bit / AWQ) |
| 7 | `06_error_analysis` | CPU, ~1 min | (HF_TOKEN to pull) | taxonomy table, chart, worked examples |
| 8 | Actions → `deploy-space` | ~5 min | HF_TOKEN | Gradio Space running the Q4 GGUF |
| 9 | Actions → `api-baseline` (any time; rerun after `test_real` lands) | ~20–30 min at 25 RPM, $0 (free tier); resumes daily when the quota runs out | repo secret GROQ_API_KEY | GPT-OSS-120B (Groq) 3-shot predictions, token counts, metrics and table, committed (resumable) |

