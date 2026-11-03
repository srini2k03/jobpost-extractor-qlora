# Run the GPU steps on your own laptop (Windows + RTX 4080)

This guide sets up a Windows laptop with an NVIDIA RTX 4080 Laptop GPU (12 GB VRAM) and 32 GB RAM to run
every GPU step of the project. The steps are the same ones the Colab notebooks run, and the outputs land in
the same places. Everything runs inside **WSL2 Ubuntu**: a Linux environment built into Windows that can see
your NVIDIA GPU.

Colab is still an option (see the README). Both paths produce identical files and push them to the same
Hugging Face repos.

**Time:**
- About 45 minutes of setup.
- About 3–4 hours of mostly unattended GPU time.
  - Paraphrase: ~30 min.
  - Baselines: ~70 min.
  - Training: ~40 min.
  - Evaluation: ~20 min.
  - AWQ: ~30 min.
  - Push: ~10 min.

**Disk:** keep about 60 GB free. The 7B teacher model is ~15 GB, plus checkpoints, merged and AWQ models,
and Python packages.

---

## 1. Update the NVIDIA driver (Windows)

1. Install the latest **NVIDIA Game Ready or Studio driver for Windows**, from the NVIDIA app or from
   nvidia.com/drivers. You need driver **570 or newer**, because the PyTorch build used here targets CUDA 12.8.
2. Reboot.

> **Do not install an NVIDIA or CUDA driver inside WSL/Ubuntu.** The Windows driver is shared with WSL
> automatically. Installing a Linux driver inside WSL breaks GPU access. You also do not need the CUDA
> Toolkit: PyTorch ships its own CUDA libraries.

## 2. Install WSL2 with Ubuntu

1. Open **PowerShell as Administrator** and run:

   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

   Ubuntu 22.04 also works: `wsl --install -d Ubuntu-22.04`.
2. Reboot when asked.
3. Open **Ubuntu** from the Start menu, then create a Linux user name and password.
4. If WSL was already installed, update it from PowerShell: `wsl --update`.

### Give WSL enough memory

By default WSL may use only half of your RAM (16 GB here). Loading and merging models goes more smoothly with
more.

1. In Windows, create the file `C:\Users\<you>\.wslconfig` with:

   ```ini
   [wsl2]
   memory=24GB
   swap=16GB
   processors=12
   ```

2. Then, in PowerShell, run `wsl --shutdown` and reopen Ubuntu.

## 3. Check that WSL sees the GPU

In the Ubuntu terminal:

```bash
nvidia-smi
```

You should see a table with **NVIDIA GeForce RTX 4080 Laptop GPU**, 12282 MiB of memory and a driver
version of 570 or higher.
- If you get "command not found" or "No devices were found", update the Windows driver, run `wsl --update`
  in PowerShell, and reboot.

## 4. Get the code and the Python tools

In Ubuntu, keep the project **inside the Linux file system** (e.g. `~/jobpost-extractor-qlora`), not under
`/mnt/c/...`. Windows-drive access from WSL is many times slower.

```bash
sudo apt update && sudo apt install -y git make build-essential curl
curl -LsSf https://astral.sh/uv/install.sh | sh      # uv: fast Python + package manager
source ~/.bashrc
git clone https://github.com/srini2k03/jobpost-extractor-qlora.git
cd jobpost-extractor-qlora
```

## 5. Create the environment (one command)

```bash
make local-setup
```

This does two things:
- creates `.venv` with Python 3.11, which uv downloads, so the Ubuntu version doesn't matter;
- installs the pinned GPU stack from [`requirements/local-gpu.txt`](../requirements/local-gpu.txt).

| package | version | why this one |
|---|---|---|
| torch | 2.8.0 (CUDA 12.8 wheel) | current stable with Ada (sm_89) and bf16; CUDA libs bundled |
| transformers | 4.56.2 | required by trl 0.23 (>= 4.56.1) |
| trl | 0.23.1 | `SFTConfig(completion_only_loss=..., max_length=...)` API used by `src/jobx/train/sft.py` |
| peft | 0.17.1 | LoRA + QLoRA, `merge_and_unload` |
| accelerate | 1.10.1 | |
| bitsandbytes | 0.48.1 | 4-bit NF4 / 8-bit on CUDA 12.x, torch >= 2.3 |
| datasets | 4.1.1 | |
| huggingface_hub | 0.35.3 | transformers 4.56 needs < 1.0 |
| wandb | 0.22.2 | |
| lm-format-enforcer | 0.11.3 | guided JSON decoding |

These are the same versions the Colab notebooks pin, plus a matching torch. AutoAWQ needs an older
transformers (4.51.3), so it gets its own environment, `.venv-awq`. `make local-awq` creates it the first
time it runs.

## 6. Log in to Hugging Face and Weights & Biases

```bash
source .venv/bin/activate
huggingface-cli login            # paste a token with *write* access (huggingface.co/settings/tokens)
wandb login                      # optional but recommended: training curves + sample generations
echo 'export HF_USERNAME=<your-hf-username>' >> ~/.bashrc && source ~/.bashrc
```

Alternatively, put `hf_username: <your-hf-username>` in `configs/hub.yaml`.

## 7. Check everything

```bash
make local-check
```

This prints a table and ends with `RESULT: PASS` or `RESULT: FAIL`. It checks:
- the Python version and that you are in WSL;
- the exact package versions above;
- that torch sees the GPU, its VRAM, and bf16 support;
- a real 4-bit bitsandbytes load and forward pass of a tiny model (built locally, no download);
- free disk space and the RAM WSL can see;
- the Hugging Face login and `HF_USERNAME`;
- the W&B login.

Fix any `FAIL` before continuing (see Troubleshooting). A `WARN` is informational.

## 8. Run the steps

Run each step on its own, in this order:

```bash
make local-paraphrase   # ~25-35 min  (notebook 01) 7B paraphrases + real-posting drafts
make local-baselines    # ~60-80 min  (notebook 02) zero/few-shot, free + guided, 1.5B and 7B
make local-train        # ~30-45 min  (notebook 03) QLoRA fine-tune, bf16
make local-eval         # ~15-20 min  (notebook 04) fine-tune predictions + merged model
make local-awq          # ~20-30 min  (notebook 05) AWQ + 16-bit / 8-bit / AWQ benchmark
make local-push         # ~5-10 min   upload everything to the Hub, exactly like the notebooks
```

Or run everything with `make local-all`, which prints a summary at the end.

Every step can be restarted:
- The paraphrase step appends each batch to `data/paraphrased/raw_generations.jsonl`.
- Baseline and eval rows that are already finished are skipped.
- Training resumes from the last `checkpoint-*`. A finished adapter is reused unless you pass
  `ARGS=--force`.
- Benchmark rows that already exist are kept.

Useful options:
- `make local-baselines ARGS="--families qwen1.5b"`: run only the 1.5B baselines.
- `make local-train ARGS="--sweep"`: also run the r/lr sweep, about 15 min per run.
- `make local-push ARGS="--dry-run"`: list what would be uploaded.

### Real test set (between paraphrase and baselines)

`local-paraphrase` fetches about 200 real postings and drafts their labels. Before running the baselines:
1. Verify the drafts with `jobx verify` (about 30 min). This creates `data/real/test_real.jsonl`.
2. Run `jobx build-data` so the paraphrases are merged into the training set.

The baselines and evaluation then also run on `test_real`, the headline split. Read
[`docs/real_dataset.md`](real_dataset.md) about the licence check first.

## 9. Where files land

These are the same paths the notebooks use:

| what | where |
|---|---|
| paraphrases + filter stats | `data/paraphrased/` |
| real-posting drafts / verified set | `data/real/test_real_draft.jsonl`, `data/real/test_real.jsonl` |
| predictions, metrics | `results/predictions/`, `results/metrics/` |
| training logs + curves | `results/training/<run>/`, `results/training/curves.png` |
| benchmark rows | `results/benchmarks/local_rtx4080.json` |
| LoRA adapter | `outputs/qwen2.5-1.5b-qlora-r16-local/final/` (checkpoints alongside) |
| merged model / AWQ model | `outputs/merged/`, `outputs/awq/` |
| downloaded base models | `~/.cache/huggingface/` (set `HF_HOME` to move it) |

`make local-push` uploads results to `<HF_USERNAME>/jobx-artifacts`, and the adapter, merged model and AWQ
model to their model repos (`configs/hub.yaml`). After that:
1. Run `jobx eval && jobx results-table`, commit `results/` and `README.md`, and push.
2. Run the **quantize-benchmark** workflow on GitHub. It builds the GGUF files from the merged model you
   pushed.

## Alternative: run the notebooks in VS Code + WSL

1. Install VS Code on Windows with the **WSL** and **Jupyter** extensions.
2. In Ubuntu, run `code .` in the repo.
3. Pick `.venv/bin/python` as the kernel and run `notebooks/0X_*.ipynb`.
4. Skip the first (Colab setup) cell. Its `pip install` line only runs on Colab, so running it locally is
   harmless too.
5. Set `HF_TOKEN` / `HF_USERNAME` / `WANDB_API_KEY` as environment variables before starting VS Code.

The notebooks call the same `jobx.pipeline` functions as the make targets. Locally they use the T4 profile
unless you change `load_profile('colab_t4')` to `load_profile('local_rtx4080')`.

## Troubleshooting

**`CUDA out of memory`**
- Close other GPU apps (browsers, games), then check `nvidia-smi`.
- Lower the batch size and keep the effective batch at 16: `make local-train ARGS="--config
  configs/train/local_rtx4080.yaml"` after editing `per_device_train_batch_size: 4` and
  `gradient_accumulation_steps: 4` in that file.
- For baselines, paraphrasing or eval, lower `batch_small` / `batch_7b` / `batch` in
  `configs/gpu/local_rtx4080.yaml`.
- Laptop GPUs share power and thermals with the CPU, so keep the laptop plugged in.

**bitsandbytes: "CUDA Setup failed" / "libcudart.so not found" / "no GPU detected"**
- Run `nvidia-smi` in WSL. If that fails, the problem is the driver or WSL, not Python.
- Check that you did *not* install `nvidia-driver-*` or `cuda` packages inside Ubuntu. If you did, remove
  them: `sudo apt remove --purge 'nvidia-*' 'cuda-*'`.
- Re-run `make local-setup` so bitsandbytes 0.48.1 and torch 2.8.0 (cu128) match.
- If `ldconfig -p | grep libcuda` shows nothing, run
  `echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc`.

**Processes get killed while loading or merging models**
- WSL ran out of RAM. Raise `memory=` in `.wslconfig` (24 GB of 32 GB is a good balance) and run
  `wsl --shutdown`.

**Disk full**
- Model downloads go to `~/.cache/huggingface`. Check with `du -sh ~/.cache/huggingface`, and remove models
  you no longer need with `huggingface-cli delete-cache`.
- Delete old checkpoints in `outputs/*/checkpoint-*` once training has finished.
- The WSL disk grows but does not shrink on its own. To compact it: run `wsl --shutdown`, then
  `Optimize-VHD` (Hyper-V) or `diskpart` → `compact vdisk` on the `ext4.vhdx`.

**Very slow file access**
- The repo is under `/mnt/c/`. Clone it into your Linux home directory instead.

**AWQ fails to install or quantise**
- AutoAWQ is archived upstream. `make local-awq` still benchmarks the 16-bit and 8-bit rows; the AWQ row
  stays "pending". Delete `.venv-awq` and re-run to rebuild the environment.

**Gated model (Llama 3.2)**
- Accept its licence on the Hugging Face model page with the same account as your token, then run
  `make local-baselines ARGS="--families llama1b"`.
