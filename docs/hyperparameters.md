# Hyper-parameters: what was chosen and why

Config: [`configs/train/qwen2.5-1.5b-qlora.yaml`](../configs/train/qwen2.5-1.5b-qlora.yaml). The sweep that
backs these choices is [`configs/train/sweep.yaml`](../configs/train/sweep.yaml), and it runs from
`notebooks/03_train_qlora.ipynb` (`RUN_SWEEP = True`). **Sweep results: pending GPU run.** Until then, the
reasons below are priors from the QLoRA literature and the T4 memory budget, not measurements on this task.

| Setting | Value | Why |
|---|---|---|
| Base precision | 4-bit NF4, double quantisation | NF4 is information-theoretically suited to normally distributed weights (QLoRA paper). Double quantisation saves ~0.4 bit/param more. A 1.5B base then needs ~1.2 GB, leaving the T4's 16 GB for activations at 1,024 tokens. |
| Compute dtype | fp16 (`fp16: true`, `bf16: false`) | The T4 (Turing, sm75) has no bf16 support. PEFT keeps the LoRA weights in fp32 (adapter autocast), and the fp16 grad scaler handles overflow. |
| LoRA rank `r` | 16 | Extraction into a fixed schema is a narrow task. r=8 is often enough, while r=32 doubles adapter size and step time for little gain on small datasets. 16 is the middle of the sweep {8, 16, 32}. |
| LoRA `alpha` | 32 (α/r = 2) | The update is scaled by α/r. Keeping α/r constant (2) across the sweep means changing r does not also change the effective learning rate, so the r comparison is fair. |
| Target modules | q, k, v, o, gate, up, down | The QLoRA paper found that adapting all linear layers is needed to match full fine-tuning. Attention-only LoRA underperforms. |
| LoRA dropout | 0.05 | Light regularisation for ~2k examples. |
| Learning rate | 2e-4, cosine, 3% warmup | The standard QLoRA adapter LR for 1–7B models. The sweep also tries 1e-4. Cosine decays to ~0 by the end, so the last checkpoints are stable. Warmup avoids early spikes from the randomly initialised B matrix interacting with fp16. |
| Batch | 4 per device × 4 accumulation = 16 | 4 × 1,024 tokens fits in T4 memory with gradient checkpointing. An effective batch of 16 gives ~135 optimiser steps per epoch on ~2.2k examples: enough steps for a smooth schedule, without the noisy updates of a batch of 4. |
| Epochs | 3, early stopping (patience 3 evals) | Small data needs a few passes. Early stopping on validation JSON micro F1 (not loss) stops when *extraction quality* plateaus, and `load_best_model_at_end` keeps the best checkpoint. |
| Max sequence length | 1,024 tokens | System prompt (~350 tokens) + posting (p95 ≈ 0.6k characters ≈ 200 tokens in the synthetic set) + JSON (~120 tokens) fit with room for longer real postings. |
| Loss | completion-only | Only the assistant JSON tokens are trained, so the model is not spending capacity on reproducing prompts. |
| Optimiser | paged AdamW 8-bit, max grad norm 0.3 | Paged optimiser states avoid OOM spikes. The 0.3 clip follows the QLoRA reference recipe. |
| Gradient checkpointing | on (non-reentrant) | It trades ~30% more step time for a large drop in activation memory, which is what lets 4 × 1,024 fit. |

## How to read the sweep

Each sweep run trains for 1 epoch and reports the best validation JSON micro F1 (`results/training/sweep_summary.json`).
Pick the smallest r within the noise of the best. If 1e-4 matches 2e-4, prefer it for stability. Record the
outcome here when it exists, replacing this sentence.

## Local RTX 4080 profile

The profile in [`configs/train/local_rtx4080.yaml`](../configs/train/local_rtx4080.yaml) inherits everything
above and changes only what the hardware calls for. The recipe and the effective batch are unchanged, so
runs are comparable.

| Setting | T4 (Colab) | RTX 4080 Laptop | Why |
|---|---|---|---|
| Compute dtype | fp16 | **bf16** | Ada supports bf16. It has fp32's exponent range, so there is no loss scaling and no fp16 overflow in attention or logits. The T4 (Turing) has no bf16. |
| `tf32` | off | **on** | TF32 tensor cores speed up the fp32 parts (LoRA weights, optimiser maths) on Ampere and newer, with negligible accuracy cost. |
| Batch × accumulation | 4 × 4 | **8 × 2** | 12 GB holds 8 × 1,024 tokens for a 4-bit 1.5B model with checkpointing. Larger micro-batches use the faster GPU better, and the effective batch stays 16, so the learning rate and schedule still apply. |
| Eval batch | 8 | 16 | Same reasoning, for the loss-only evaluation pass. |
| `dataloader_num_workers` | 0 | **2** | Under WSL2 a couple of workers keep tokenised batches ready. More just compete with the GPU feeder for CPU and RAM. |
| Sequence length | 1,024 | 1,024 | Unchanged. |
| Gradient checkpointing | on | on | Kept for headroom with long real postings and the in-loop generations used for JSON-F1. |
| Optimiser | paged AdamW 8-bit | paged AdamW 8-bit | Unchanged. Paged states avoid OOM spikes on 12 GB. |
| Resume | - | `resume: true` | Continues from the last `checkpoint-*` after an interruption. Laptops sleep. |

## CPU route

This is the route that is actually run. Configuration: [`configs/cpu/route.yaml`](../configs/cpu/route.yaml).
It runs on GitHub-hosted `ubuntu-latest` runners (4 vCPU, 16 GB RAM, 6 h per job; usually AMD EPYC with
AVX2).

| Setting | Value | Why |
|---|---|---|
| Base model | Qwen2.5-0.5B-Instruct | Ungated: it downloads in Actions without a token. At ~0.5B parameters, an epoch over ~2.2k short examples takes hours on 4 CPU cores, not days. The Q4_K_M GGUF (~0.4 GB) is small enough to run in a browser. |
| Prompt | compact system prompt | The fine-tune learns the labelling conventions from the data, so the long rule list (~400 tokens) is dead weight. A one-line instruction cuts sequences from ~690 to ~330 tokens (median, estimated), roughly halving training compute and in-browser prompt time. Baselines keep the full prompt; they need the rules. |
| `max_length` | 512 (from measured p99) | The first job tokenises every training example and picks 512 if p99 fits, else p99 rounded up to 64 (cap 768). Longer examples are **dropped, not truncated**: cutting the prompt changes the task, and cutting the completion teaches broken JSON. The count is recorded in `plan.json` and the model card. |
| LoRA | r=16, α=32, dropout 0.05, all linear projections | Same as the GPU route, so the two are comparable (see above for why). |
| Precision | fp32 (bf16 only with AVX512-BF16/AMX) | Without native bf16 instructions, bf16 on CPU is emulated and slower than fp32. The planner detects the CPU flags. |
| Batch | 2 × 8 accumulation = 16 | Same effective batch as the GPU route. In a local test on a 4-core, 16 GB machine with the same architecture, a micro-batch of 8 × 360 tokens was OOM-killed at ~14 GB RSS, and 4 × 360 peaked at ~10.6 GB. The fp32 logits over Qwen's 152k-token vocabulary dominate memory. A micro-batch of 2 leaves headroom on the runner's 16 GB, at almost the same speed per example. |
| Learning rate | 3e-4, cosine, 5% warmup | Slightly higher than the GPU route's 2e-4: the run has fewer optimiser steps (~130–270) and a smaller model, which tolerates it. Warmup is a bit longer for the same reason. |
| Gradient checkpointing | off | RAM is not the constraint at 0.5B. Checkpointing would cost ~30% more compute. |
| Epochs / data | planned from a measured benchmark | The first job times 4 optimiser steps (the first is discarded) and picks 2 epochs, else 1 epoch, else a **stratified subset** (proportional over template family × domain) that fits 3 chained jobs × 300 min × 85% (the rest is evals and saves). The choice and its reason are written to `results/training/qwen2.5-0.5b-lora-cpu/plan.json`. |
| Evaluation | 4 times per run: loss on 64 val examples, JSON-F1 on 24 | Greedy generation on CPU is slow, so the JSON-F1 subset is small. It is a trend signal; the real evaluation is the full test split in `cpu_eval_release`. |
| Threads | all cores (`torch.set_num_threads(nproc)`) | |

**Measured locally before the first CI run.** I ran a timing test with the exact Qwen2.5-0.5B architecture
(494M parameters, random weights: speed and memory only) and LoRA r=16 on all linear layers, in fp32. The
machine had 4 threads (Intel Xeon @ 2.8 GHz) and 16 GB RAM, the same size as a GitHub runner. With
micro-batches of 2 × 360 tokens and 8 accumulation steps:
- **~108 s per optimiser step of 16** (~13.5 s per micro-batch, ~6.7 s per example);
- peak RSS ~9.9 GB.

Implications:
- One epoch over ~2.2k examples is ~137 steps ≈ 4.1 h. Two epochs, ~8.2 h, fit the 12.75 h budget, so the
  planner should pick **2 epochs over all data in 2–3 chained jobs** (each job trains ~4.25 h).
- A slower runner gets 1 epoch or a subset instead.
- The first CI job re-measures on the runner's own CPU, and `plan.json` records what was actually used.
