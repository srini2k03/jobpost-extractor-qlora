"""Generate the Colab notebooks in notebooks/ (single source of truth for their setup cells).

    python scripts/build_notebooks.py

Notebooks hold orchestration only; all logic is imported from src/jobx.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
GH = "srini2k03/jobpost-extractor-qlora"
REPO_URL = f"https://github.com/{GH}.git"

# Pinned for a Colab T4 runtime (fp16, no bf16). torch/CUDA come from the Colab image.
GPU_PINS = (
    "transformers==4.56.2 peft==0.17.1 trl==0.23.1 bitsandbytes==0.48.1 accelerate==1.10.1 "
    "datasets==4.1.1 huggingface_hub==0.35.3 wandb==0.22.2 lm-format-enforcer==0.11.3"
)
# AutoAWQ is archived upstream and pins an older transformers; it gets its own runtime.
AWQ_PINS = "autoawq==0.2.9 transformers==4.51.3 accelerate==1.10.1 huggingface_hub==0.35.3 datasets==4.1.1"


def badge(name: str) -> str:
    url = f"https://colab.research.google.com/github/{GH}/blob/main/notebooks/{name}.ipynb"
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


def intro(name: str, title: str, what: str, runtime: str, hw: str, secrets: list[str], outputs: str) -> str:
    sec = ", ".join(f"`{s}`" for s in secrets) if secrets else "none"
    return f"""{badge(name)}

# {title}

{what}

**How to run**
1. Runtime → Change runtime type → **{hw}**.
2. Secrets (key icon in the left sidebar), each with *Notebook access* switched on: {sec}.
   `HF_USERNAME` is your Hugging Face account name; `HF_TOKEN` needs *write* access.
3. Runtime → **Run all**. Expected runtime: **{runtime}**. Cells after the setup are safe to re-run.

**Outputs:** {outputs}

All code lives in `src/jobx` of the GitHub repo; this notebook only orchestrates it."""


def setup_cell(pins: str | None, gpu: bool = True) -> str:
    pip = f"!pip install -q {pins}\n" if pins else ""
    check = (
        (
            "import torch\n"
            "assert torch.cuda.is_available(), 'No GPU: Runtime -> Change runtime type -> T4 GPU'\n"
            "print(torch.__version__, torch.cuda.get_device_name(0))\n"
        )
        if gpu
        else ""
    )
    return f"""# Setup: clone the repo (code + data), install pinned dependencies, install jobx itself.
import os, sys
IN_COLAB = "google.colab" in sys.modules
REPO_URL = "{REPO_URL}"
if IN_COLAB:
    if not os.path.exists("/content/jobpost-extractor-qlora"):
        !git clone --depth 1 {{REPO_URL}} /content/jobpost-extractor-qlora
    %cd /content/jobpost-extractor-qlora
    {pip.strip() or "pass"}
    !pip install -q -e .
else:
    # running locally from a checkout: notebooks/ -> repo root
    if os.path.basename(os.getcwd()) == "notebooks":
        os.chdir("..")
{check}"""


SECRETS_CELL = """# Secrets -> environment variables (read by jobx.hub and wandb).
import os
for key in ("HF_TOKEN", "HF_USERNAME", "WANDB_API_KEY"):
    if not os.environ.get(key):
        try:
            from google.colab import userdata
            os.environ[key] = userdata.get(key)
        except Exception:
            print(f"secret {key} not available")
from jobx.hub import hf_username, repo_id
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("Hub account:", hf_username(), "| artifacts repo:", repo_id("artifacts"))"""


def nb(name: str, cells: list[tuple[str, str]]) -> nbformat.NotebookNode:
    n = new_notebook()
    n.metadata = {
        "accelerator": "GPU" if name != "06_error_analysis" else "None",
        "colab": {"provenance": [], "gpuType": "T4"} if name != "06_error_analysis" else {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    n.cells = [new_markdown_cell(src) if kind == "md" else new_code_cell(src) for kind, src in cells]
    for i, c in enumerate(n.cells):
        c.id = f"{name[:2]}-{i:02d}"  # deterministic ids so regenerating gives identical files
    return n


def nb01() -> nbformat.NotebookNode:
    name = "01_paraphrase_7b"
    return nb(
        name,
        [
            (
                "md",
                intro(
                    name,
                    "01 · Paraphrase pass (7B) + real test set drafts",
                    "Rewrites a subset of synthetic training postings with **Qwen2.5-7B-Instruct (4-bit)** for "
                    "realism, keeps only rewrites that pass the rule + NLI checker, then fetches ~200 real "
                    "public postings and drafts their labels with the same 7B model (few-shot) for human "
                    "verification with `jobx verify`.",
                    "~1.5 h (paraphrase ~60 min for 600 postings, drafts ~20 min)",
                    "T4 GPU",
                    ["HF_TOKEN", "HF_USERNAME"],
                    "`data/paraphrased/paraphrased_train.jsonl` + `filter_stats.json`, "
                    "`data/real/test_real_draft.jsonl` (+ `raw_meta.json`), all pushed to the private "
                    "dataset repo `<HF_USERNAME>/jobx-artifacts`.",
                ),
            ),
            ("code", setup_cell(GPU_PINS)),
            ("code", SECRETS_CELL),
            ("md", "## Settings"),
            (
                "code",
                "N_PARAPHRASE = 600      # training postings to rewrite\n"
                "BATCH = 8\n"
                "TEACHER = 'Qwen/Qwen2.5-7B-Instruct'\n"
                "NLI_MODEL = 'cross-encoder/nli-deberta-v3-small'\n"
                "DO_REAL_DRAFTS = True",
            ),
            ("md", "## 1. Paraphrase a training subset"),
            (
                "code",
                "import json, random\n"
                "from pathlib import Path\n"
                "from jobx.data.build import read_jsonl, write_jsonl\n"
                "from jobx.data.paraphrase import STYLES, paraphrase_messages, pick_subset, filter_paraphrases\n"
                "from jobx.baselines.hf_runner import load_model, generate\n"
                "\n"
                "train = read_jsonl('data/processed/train.jsonl')\n"
                "subset = pick_subset(train, N_PARAPHRASE)\n"
                "rng = random.Random(0)\n"
                "styles = [rng.choice(STYLES) for _ in subset]\n"
                "model, tok = load_model(TEACHER, quant='4bit')\n"
                "outs = []\n"
                "from tqdm.auto import tqdm\n"
                "for i in tqdm(range(0, len(subset), BATCH)):\n"
                "    chunk = subset[i:i + BATCH]\n"
                "    msgs = [paraphrase_messages(e['text'], s) for e, s in zip(chunk, styles[i:i + BATCH])]\n"
                "    outs += [g.text for g in generate(model, tok, msgs, max_new_tokens=700, sample=True)]\n"
                "len(outs)",
            ),
            ("md", "## 2. Validate: every gold value recoverable, nothing new (rules), entailment (NLI)"),
            (
                "code",
                "from jobx.data.validate import hf_nli_scorer\n"
                "nli = hf_nli_scorer(NLI_MODEL, device=0)\n"
                "kept, stats = filter_paraphrases(list(zip(subset, styles, outs)), nli=nli, nli_threshold=0.5)\n"
                "Path('data/paraphrased').mkdir(parents=True, exist_ok=True)\n"
                "write_jsonl('data/paraphrased/paraphrased_train.jsonl', kept)\n"
                "stats['teacher'] = TEACHER; stats['nli_model'] = NLI_MODEL\n"
                "Path('data/paraphrased/filter_stats.json').write_text(json.dumps(stats, indent=2))\n"
                "print(json.dumps(stats, indent=2))\n"
                "for e in kept[:2]:\n"
                "    print('-' * 80); print(e['text'])",
            ),
            (
                "code",
                "from jobx.baselines.runs import push_results\n"
                "push_results([Path('data/paraphrased/paraphrased_train.jsonl').resolve(),\n"
                "              Path('data/paraphrased/filter_stats.json').resolve()])",
            ),
            (
                "md",
                "## 3. Real postings: fetch + draft labels\n"
                "The fetch picks the first candidate in `configs/real_data.yaml` whose declared licence is in "
                "the allowlist. **A human still has to confirm the licence** (see `docs/real_dataset.md`). "
                "Drafts go to a *private* dataset repo; verify them locally with `jobx verify`.",
            ),
            (
                "code",
                "import subprocess\n"
                "if DO_REAL_DRAFTS:\n"
                "    r = subprocess.run(['jobx', 'fetch-real'], capture_output=True, text=True)\n"
                "    print(r.stdout[-3000:], r.stderr[-3000:])",
            ),
            (
                "code",
                "from jobx import prompts\n"
                "from jobx.schema import parse_output, JobPosting\n"
                "raw_path = Path('data/real/raw_postings.jsonl')\n"
                "if DO_REAL_DRAFTS and raw_path.exists():\n"
                "    raw = read_jsonl(raw_path)\n"
                "    shots = prompts.select_few_shot(train, k=3)\n"
                "    drafts = []\n"
                "    for i in tqdm(range(0, len(raw), 4)):\n"
                "        chunk = raw[i:i + 4]\n"
                "        gens = generate(model, tok, [prompts.messages(r['text'], shots) for r in chunk], max_new_tokens=400)\n"
                "        for r, g in zip(chunk, gens):\n"
                "            pr = parse_output(g.text)\n"
                "            label = pr.posting.to_json_dict() if pr.posting else JobPosting().to_json_dict()\n"
                "            drafts.append({'id': r['id'], 'text': r['text'], 'label': label,\n"
                "                           'meta': {'source': 'real', 'dataset': r['dataset'], 'row': r['row'],\n"
                "                                    'licence': r['licence'], 'draft_model': TEACHER,\n"
                "                                    'draft_schema_ok': pr.schema_ok}})\n"
                "    write_jsonl('data/real/test_real_draft.jsonl', drafts)\n"
                "    print(sum(d['meta']['draft_schema_ok'] for d in drafts), 'of', len(drafts), 'drafts schema-valid')\n"
                "    push_results([Path('data/real/test_real_draft.jsonl').resolve(), Path('data/real/raw_meta.json').resolve()])\n"
                "else:\n"
                "    print('no real postings fetched; see the output above')",
            ),
            (
                "md",
                "## Next\nOn your machine: `jobx pull` (downloads these files), `jobx verify` (≈30 min for a "
                "sample), then `jobx build-data` to merge the paraphrases and refresh `DATA_CARD.md`; commit "
                "and push. Then run `02_baselines` and `03_train_qlora`.",
            ),
        ],
    )


def nb02() -> nbformat.NotebookNode:
    name = "02_baselines"
    return nb(
        name,
        [
            (
                "md",
                intro(
                    name,
                    "02 · Prompting baselines (zero-/few-shot, free vs guided JSON)",
                    "Runs the base models with the **same system prompt and schema** used for fine-tuning: "
                    "Qwen2.5-1.5B-Instruct zero-shot and 3-shot, the optional Llama-3.2-1B-Instruct 3-shot, and "
                    "the **Qwen2.5-7B-Instruct 4-bit 3-shot upper bound**. Each runs with free decoding and with "
                    "**guided JSON decoding** (lm-format-enforcer), to show what decoding constraints alone buy.",
                    "~2–2.5 h in total; each section saves and pushes its own results, so you can split it "
                    "across sessions with the `RUN_*` flags",
                    "T4 GPU",
                    ["HF_TOKEN", "HF_USERNAME"],
                    "`results/predictions/<approach>__<split>.jsonl` + `.meta.json` and "
                    "`results/metrics/<approach>__<split>.json` for every row, pushed to `<HF_USERNAME>/jobx-artifacts`.",
                ),
            ),
            ("code", setup_cell(GPU_PINS)),
            ("code", SECRETS_CELL),
            (
                "md",
                "## Settings\nGuided decoding runs one posting at a time, so it uses a fixed, evenly spaced "
                "subset (`GUIDED_MAX`) that is identical for every approach; the table reports `n`.",
            ),
            (
                "code",
                "RUN_QWEN_1_5B = True\n"
                "RUN_LLAMA_1B = False   # gated model: accept the licence on its HF page first\n"
                "RUN_QWEN_7B = True\n"
                "GUIDED_MAX = 200\n"
                "FREE_MAX = None        # None = whole split\n"
                "from jobx.baselines.runs import run_approach, available_test_splits\n"
                "from jobx.baselines.hf_runner import load_model\n"
                "from jobx.data.build import read_jsonl\n"
                "from jobx import prompts\n"
                "import gc, torch\n"
                "SPLITS = available_test_splits()\n"
                "SHOTS = prompts.select_few_shot(read_jsonl('data/processed/train.jsonl'), k=3)\n"
                "print('splits:', SPLITS, '| shots:', [s['id'] for s in SHOTS])",
            ),
            (
                "code",
                "def run_family(model_id, prefix, quant=None, zero_shot=True, batch=16):\n"
                "    model, tok = load_model(model_id, quant=quant)\n"
                "    for split in SPLITS:\n"
                "        if zero_shot:\n"
                "            run_approach(model, tok, f'{prefix}_zero_shot', split, model_id=model_id, max_examples=FREE_MAX, batch_size=batch)\n"
                "            run_approach(model, tok, f'{prefix}_zero_shot_guided', split, model_id=model_id, guided=True, max_examples=GUIDED_MAX)\n"
                "        run_approach(model, tok, f'{prefix}_few_shot', split, model_id=model_id, shots=SHOTS, max_examples=FREE_MAX, batch_size=max(1, batch // 2))\n"
                "        run_approach(model, tok, f'{prefix}_few_shot_guided', split, model_id=model_id, shots=SHOTS, guided=True, max_examples=GUIDED_MAX)\n"
                "    del model; gc.collect(); torch.cuda.empty_cache()",
            ),
            ("md", "## Qwen2.5-1.5B-Instruct (fp16), ~45 min"),
            ("code", "if RUN_QWEN_1_5B:\n    run_family('Qwen/Qwen2.5-1.5B-Instruct', 'qwen2.5-1.5b')"),
            ("md", "## Llama-3.2-1B-Instruct (fp16, optional), ~30 min"),
            (
                "code",
                "if RUN_LLAMA_1B:\n    run_family('meta-llama/Llama-3.2-1B-Instruct', 'llama3.2-1b', zero_shot=False)",
            ),
            ("md", "## Upper bound: Qwen2.5-7B-Instruct 3-shot, 4-bit NF4, ~75 min"),
            (
                "code",
                "if RUN_QWEN_7B:\n    run_family('Qwen/Qwen2.5-7B-Instruct', 'qwen2.5-7b', quant='4bit', zero_shot=False, batch=8)",
            ),
            ("md", "## Summary"),
            (
                "code",
                "from jobx.eval.results_table import build_tables\n"
                "from IPython.display import Markdown\n"
                "Markdown(build_tables('.')[0])",
            ),
        ],
    )


def nb03() -> nbformat.NotebookNode:
    name = "03_train_qlora"
    return nb(
        name,
        [
            (
                "md",
                intro(
                    name,
                    "03 · QLoRA fine-tuning (TRL SFTTrainer)",
                    "Fine-tunes **Qwen2.5-1.5B-Instruct** with QLoRA (4-bit NF4, double quantisation, fp16 compute), "
                    "LoRA r=16/α=32 on all linear projections, completion-only loss, gradient checkpointing, "
                    "cosine LR 2e-4 with 3% warmup, effective batch 16, early stopping on validation JSON micro F1. "
                    "Weights & Biases logs loss, LR, eval metrics and sample generations. Optional: the small "
                    "r/lr sweep behind `docs/hyperparameters.md`, and the Llama-3.2-1B comparison run.",
                    "~1–1.5 h for the main run; sweep +~25 min per run",
                    "T4 GPU",
                    ["HF_TOKEN", "HF_USERNAME", "WANDB_API_KEY"],
                    "LoRA adapter pushed to `<HF_USERNAME>/jobx-qwen2.5-1.5b-qlora` (with model card); "
                    "`results/training/<run>/log_history.json`, `best.json`, `curves.png` pushed to "
                    "`<HF_USERNAME>/jobx-artifacts`; the W&B run.",
                ),
            ),
            ("code", setup_cell(GPU_PINS)),
            ("code", SECRETS_CELL),
            (
                "code",
                "CONFIG = 'configs/train/qwen2.5-1.5b-qlora.yaml'\n"
                "RUN_SWEEP = False\n"
                "RUN_LLAMA = False   # gated: accept the Llama 3.2 licence on HF first\n"
                "import json\n"
                "from pathlib import Path\n"
                "from jobx.train.config import load_train_config, sweep_runs\n"
                "from jobx.train.sft import train\n"
                "cfg = load_train_config(CONFIG)\n"
                "print(json.dumps(cfg, indent=2))",
            ),
            ("md", "## Train"),
            ("code", "final = train(cfg)\nprint(Path(final, 'best.json').read_text())"),
            ("md", "## Curves + push artefacts"),
            (
                "code",
                "import os, shutil\n"
                "from jobx.eval.curves import plot_curves\n"
                "from jobx.baselines.runs import push_results\n"
                "from jobx.cards import render_card, metrics_table\n"
                "from jobx.hub import repo_id, push_folder\n"
                "run_dir = Path('results/training') / cfg['run_name']\n"
                "run_dir.mkdir(parents=True, exist_ok=True)\n"
                "for f in ('log_history.json', 'best.json', 'train_config.json'):\n"
                "    shutil.copy(Path(final, f), run_dir / f)\n"
                "png = plot_curves(run_dir / 'log_history.json', run_dir / 'curves.png')\n"
                "shutil.copy(png, 'results/training/curves.png')\n"
                "push_results([run_dir / 'log_history.json', run_dir / 'best.json', run_dir / 'train_config.json', png.resolve(), Path('results/training/curves.png').resolve()])\n"
                "import wandb\n"
                "card = render_card('adapter', {\n"
                "    'base_model': cfg['base_model'], 'repo_name': repo_id('adapter'), 'artifacts_repo': repo_id('artifacts'),\n"
                "    'n_train': sum(1 for _ in open(cfg['data']['train'])), 'lora_r': cfg['lora']['r'], 'lora_alpha': cfg['lora']['alpha'],\n"
                "    'learning_rate': cfg['training']['learning_rate'], 'wandb_url': os.environ.get('WANDB_RUN_URL', 'see W&B project jobx-qlora'),\n"
                "    'metrics_table': metrics_table(['qwen2.5-1.5b_qlora']), 'base_licence': 'Apache-2.0'})\n"
                "Path(final, 'README.md').write_text(card)\n"
                "print('pushed adapter to', push_folder(final, 'adapter', commit_message='QLoRA adapter'))",
            ),
            ("md", "## Optional: hyper-parameter sweep (r ∈ {8,16,32}, lr ∈ {1e-4, 2e-4}; 1 epoch each)"),
            (
                "code",
                "if RUN_SWEEP:\n"
                "    summary = {}\n"
                "    for name, scfg in sweep_runs('configs/train/sweep.yaml'):\n"
                "        out = train(scfg)\n"
                "        summary[name] = json.loads(Path(out, 'best.json').read_text())\n"
                "        print(name, summary[name])\n"
                "    Path('results/training/sweep_summary.json').write_text(json.dumps(summary, indent=2))\n"
                "    push_results([Path('results/training/sweep_summary.json').resolve()])",
            ),
            ("md", "## Optional: Llama-3.2-1B-Instruct with the same recipe"),
            (
                "code",
                "if RUN_LLAMA:\n"
                "    lcfg = load_train_config('configs/train/llama3.2-1b-qlora.yaml')\n"
                "    lfinal = train(lcfg)\n"
                "    print('pushed', push_folder(lfinal, 'adapter_llama', commit_message='QLoRA adapter (Llama 3.2 1B)'))",
            ),
        ],
    )


def nb04() -> nbformat.NotebookNode:
    name = "04_eval_predict"
    return nb(
        name,
        [
            (
                "md",
                intro(
                    name,
                    "04 · Fine-tuned model: predictions + merge",
                    "Loads the base model in fp16 with the pushed LoRA adapter, writes predictions on every test "
                    "split (free decoding on the full split, guided decoding on the shared subset), then merges "
                    "the adapter and pushes the merged fp16 model that the GGUF/AWQ steps quantise.",
                    "~45–60 min",
                    "T4 GPU",
                    ["HF_TOKEN", "HF_USERNAME"],
                    "predictions + metrics for `qwen2.5-1.5b_qlora` and `qwen2.5-1.5b_qlora_guided` pushed to "
                    "`<HF_USERNAME>/jobx-artifacts`; merged model pushed to `<HF_USERNAME>/jobx-qwen2.5-1.5b-merged`.",
                ),
            ),
            ("code", setup_cell(GPU_PINS)),
            ("code", SECRETS_CELL),
            (
                "code",
                "BASE = 'Qwen/Qwen2.5-1.5B-Instruct'\n"
                "GUIDED_MAX = 200\n"
                "RUN_LLAMA = False\n"
                "from pathlib import Path\n"
                "from jobx.baselines.hf_runner import load_model\n"
                "from jobx.baselines.runs import run_approach, available_test_splits\n"
                "from jobx.hub import repo_id\n"
                "ADAPTER = repo_id('adapter')\n"
                "SPLITS = available_test_splits()\n"
                "model, tok = load_model(BASE, adapter=ADAPTER, merge_adapter=True)",
            ),
            ("md", "## Predictions"),
            (
                "code",
                "for split in SPLITS:\n"
                "    run_approach(model, tok, 'qwen2.5-1.5b_qlora', split, model_id=ADAPTER, batch_size=16)\n"
                "    run_approach(model, tok, 'qwen2.5-1.5b_qlora_guided', split, model_id=ADAPTER, guided=True, max_examples=GUIDED_MAX)",
            ),
            ("md", "## Merge + push the fp16 model"),
            (
                "code",
                "import gc, torch\n"
                "del model; gc.collect(); torch.cuda.empty_cache()\n"
                "from jobx.train.merge import merge\n"
                "from jobx.cards import render_card, metrics_table\n"
                "from jobx.hub import push_folder\n"
                "out = merge(BASE, ADAPTER, 'outputs/merged')\n"
                "card = render_card('merged', {'base_model': BASE, 'adapter_repo': ADAPTER, 'repo_name': repo_id('merged'),\n"
                "    'artifacts_repo': repo_id('artifacts'), 'n_train': sum(1 for _ in open('data/processed/train.jsonl')),\n"
                "    'lora_r': 16, 'lora_alpha': 32, 'learning_rate': '2e-4', 'wandb_url': 'see W&B project jobx-qlora',\n"
                "    'metrics_table': metrics_table(['qwen2.5-1.5b_qlora', 'qwen2.5-1.5b_qlora_guided']), 'base_licence': 'Apache-2.0'})\n"
                "Path(out, 'README.md').write_text(card)\n"
                "print('pushed', push_folder(out, 'merged', commit_message='merged fp16'))",
            ),
            ("md", "## Optional: Llama-3.2-1B fine-tune predictions"),
            (
                "code",
                "if RUN_LLAMA:\n"
                "    lmodel, ltok = load_model('meta-llama/Llama-3.2-1B-Instruct', adapter=repo_id('adapter_llama'), merge_adapter=True)\n"
                "    for split in SPLITS:\n"
                "        run_approach(lmodel, ltok, 'llama3.2-1b_qlora', split, model_id=repo_id('adapter_llama'), batch_size=16)",
            ),
            (
                "md",
                "## Next\nGitHub → Actions → **quantize-benchmark** → Run workflow (GGUF Q8_0/Q4_K_M on CPU), "
                "and `05_awq_benchmark` for the T4 rows.",
            ),
        ],
    )


def nb05() -> nbformat.NotebookNode:
    name = "05_awq_benchmark"
    return nb(
        name,
        [
            (
                "md",
                intro(
                    name,
                    "05 · AWQ 4-bit + T4 benchmark (fp16 vs 8-bit vs AWQ)",
                    "Quantises the merged model with AutoAWQ (4-bit GEMM, group 128, calibrated on *training* "
                    "postings) and benchmarks fp16, bitsandbytes 8-bit and AWQ on the same evenly spaced test "
                    "subset at batch size 1: latency, tokens/s, peak VRAM, size on disk and accuracy. "
                    "AutoAWQ is archived upstream, so it gets its own pinned environment; if it fails to install "
                    "on your Colab image, the AWQ row stays *pending* and the other rows still run.",
                    "~45 min",
                    "T4 GPU",
                    ["HF_TOKEN", "HF_USERNAME"],
                    "`results/benchmarks/t4.json`, predictions + metrics for `qwen2.5-1.5b_merged_int8` and "
                    "`qwen2.5-1.5b_awq` (pushed to `<HF_USERNAME>/jobx-artifacts`); AWQ model pushed to "
                    "`<HF_USERNAME>/jobx-qwen2.5-1.5b-awq`.",
                ),
            ),
            ("code", setup_cell(AWQ_PINS + " bitsandbytes==0.48.1")),
            ("code", SECRETS_CELL),
            (
                "code",
                "N_BENCH = 150\n"
                "SPLIT = 'test_real' if os.path.exists('data/real/test_real.jsonl') else 'test_synth'\n"
                "import json, gc, torch\n"
                "from pathlib import Path\n"
                "from huggingface_hub import snapshot_download\n"
                "from jobx.data.build import read_jsonl\n"
                "from jobx.eval.run import SPLIT_FILES, eval_subset, git_commit\n"
                "from jobx.baselines.hf_runner import load_model\n"
                "from jobx.baselines.runs import push_results\n"
                "from jobx.quantize.gpu_bench import bench_transformers, dir_size_gb, awq_quantize\n"
                "from jobx.hub import repo_id, push_folder\n"
                "MERGED = snapshot_download(repo_id('merged'))\n"
                "EX = eval_subset(read_jsonl(SPLIT_FILES[SPLIT]), N_BENCH)\n"
                "META = {'hardware': 'T4', 'commit': git_commit(), 'split': SPLIT}\n"
                "rows = []",
            ),
            (
                "code",
                "def bench(variant, approach, quant=None, path=MERGED, size=None):\n"
                "    model, tok = load_model(path, quant=quant)\n"
                "    row = bench_transformers(model, tok, EX, variant=variant, approach=approach, split=SPLIT, size_gb=size,\n"
                "                             meta={**META, 'model': path})\n"
                "    print(row); rows.append(row)\n"
                "    del model; gc.collect(); torch.cuda.empty_cache()\n"
                "fp16_gb = dir_size_gb(MERGED)\n"
                "bench('fp16 (transformers)', 'qwen2.5-1.5b_merged_fp16_bench', size=fp16_gb)\n"
                "bench('merged 8-bit (bitsandbytes)', 'qwen2.5-1.5b_merged_int8', quant='8bit', size=round(fp16_gb / 2, 3))",
            ),
            ("md", "## AWQ"),
            (
                "code",
                "try:\n"
                "    calib = [e['text'] for e in read_jsonl('data/processed/train.jsonl')[:128]]\n"
                "    awq_dir = awq_quantize(MERGED, 'outputs/awq', calib)\n"
                "    from awq import AutoAWQForCausalLM\n"
                "    from transformers import AutoTokenizer\n"
                "    amodel = AutoAWQForCausalLM.from_quantized(str(awq_dir), fuse_layers=False)\n"
                "    atok = AutoTokenizer.from_pretrained(str(awq_dir)); atok.padding_side = 'left'\n"
                "    row = bench_transformers(amodel.model, atok, EX, variant='AWQ 4-bit', approach='qwen2.5-1.5b_awq',\n"
                "                             split=SPLIT, size_gb=dir_size_gb(awq_dir), meta={**META, 'model': 'awq'})\n"
                "    print(row); rows.append(row)\n"
                "    print('pushed', push_folder(awq_dir, 'awq', commit_message='AWQ 4-bit'))\n"
                "except Exception as e:\n"
                "    print('AWQ step failed, row stays pending:', repr(e))",
            ),
            (
                "code",
                "Path('results/benchmarks').mkdir(parents=True, exist_ok=True)\n"
                "Path('results/benchmarks/t4.json').write_text(json.dumps({'rows': rows}, indent=2))\n"
                "files = [Path('results/benchmarks/t4.json').resolve()]\n"
                "for a in ('qwen2.5-1.5b_merged_int8', 'qwen2.5-1.5b_awq', 'qwen2.5-1.5b_merged_fp16_bench'):\n"
                "    for suffix in ('.jsonl', '.meta.json'):\n"
                "        p = Path(f'results/predictions/{a}__{SPLIT}{suffix}')\n"
                "        if p.exists(): files.append(p.resolve())\n"
                "push_results(files)\n"
                "rows",
            ),
        ],
    )


def nb06() -> nbformat.NotebookNode:
    name = "06_error_analysis"
    return nb(
        name,
        [
            (
                "md",
                intro(
                    name,
                    "06 · Error analysis (CPU)",
                    "Automatic failure taxonomy for every prediction file in `results/predictions`: invalid JSON, "
                    "truncation, schema-invalid, wrong salary unit, missed/spurious skill, hallucinated field, "
                    "location confusion, seniority mismatch, experience/role/company errors, with counts per "
                    "approach and worked examples. Runs on CPU from saved predictions, locally or in Colab.",
                    "~1 min",
                    "CPU (no GPU needed)",
                    ["HF_TOKEN (only to pull private artefacts)"],
                    "`results/error_taxonomy.json` and `results/error_taxonomy.png`.",
                ),
            ),
            ("code", setup_cell(None, gpu=False)),
            (
                "code",
                "# Optional: pull the latest GPU-run predictions from the HF artifacts repo.\n"
                "PULL = False\n"
                "if PULL:\n"
                "    from jobx.hub import pull_artifacts\n"
                "    pull_artifacts('.')",
            ),
            (
                "code",
                "import json\n"
                "from pathlib import Path\n"
                "import pandas as pd\n"
                "from jobx.eval.run import SPLIT_FILES, parse_pred_name, score_file, load_gold\n"
                "from jobx.eval.taxonomy import CATEGORIES, taxonomy_counts, worked_examples\n"
                "scores, golds = {}, {}\n"
                "for p in sorted(Path('results/predictions').glob('*__*.jsonl')):\n"
                "    approach, split = parse_pred_name(p)\n"
                "    if not Path(SPLIT_FILES.get(split, '-')).exists():\n"
                "        continue\n"
                "    scores[p.stem], _ = score_file(p)\n"
                "    golds[p.stem] = load_gold(split)\n"
                "counts = {k: taxonomy_counts(v) for k, v in scores.items()}\n"
                "df = pd.DataFrame(counts).T\n"
                "rates = (df[list(CATEGORIES)].div(df['n_examples'], axis=0) * 100).round(1)\n"
                "Path('results/error_taxonomy.json').write_text(json.dumps(counts, indent=2))\n"
                "rates",
            ),
            (
                "md",
                "Rates are % of examples with at least one error of that type (an example can have several).",
            ),
            (
                "code",
                "import matplotlib\n"
                "import matplotlib.pyplot as plt\n"
                "ax = rates.T.plot(kind='barh', figsize=(9, 6))\n"
                "ax.set_xlabel('% of examples'); ax.invert_yaxis(); plt.tight_layout()\n"
                "plt.savefig('results/error_taxonomy.png', dpi=120)",
            ),
            ("md", "## Per-field exact match"),
            (
                "code",
                "from jobx.eval.metrics import aggregate\n"
                "pd.DataFrame({k: {f: round(100 * v['em'], 1) for f, v in aggregate(s, n_boot=0)['per_field'].items()}\n"
                "              for k, s in scores.items()}).T",
            ),
            ("md", "## Worked examples per category"),
            (
                "code",
                "for name, s in scores.items():\n"
                "    for cat in CATEGORIES:\n"
                "        ex = worked_examples(s, golds[name], cat, k=1)\n"
                "        if not ex:\n"
                "            continue\n"
                "        e = ex[0]\n"
                "        print(f'=== {name} | {cat} | {e[\"id\"]}')\n"
                "        print(e['text'][:400].replace(chr(10), ' / '))\n"
                "        for f, d in e['diffs'].items():\n"
                '            print(f\'   {f}: gold={d["gold"]!r} pred={d["pred"]!r}\')',
            ),
        ],
    )


def main() -> None:
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for fn in (nb01, nb02, nb03, nb04, nb05, nb06):
        n = fn()
        name = fn.__name__
        fname = {
            "nb01": "01_paraphrase_7b",
            "nb02": "02_baselines",
            "nb03": "03_train_qlora",
            "nb04": "04_eval_predict",
            "nb05": "05_awq_benchmark",
            "nb06": "06_error_analysis",
        }[name]
        nbformat.validate(n)
        nbformat.write(n, out / f"{fname}.ipynb")
        print("wrote", fname)


if __name__ == "__main__":
    main()
