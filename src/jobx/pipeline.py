"""GPU pipeline steps shared by the Colab notebooks and the local-GPU scripts (``python -m jobx.local``).

Every function here is what a notebook section does, parameterised by a GPU profile
(``configs/gpu/colab_t4.yaml`` or ``configs/gpu/local_rtx4080.yaml``) so both paths produce the same files:

* ``paraphrase``      -> data/paraphrased/{raw_generations,paraphrased_train}.jsonl, filter_stats.json,
                         data/real/test_real_draft.jsonl
* ``baselines``       -> results/predictions|metrics/<approach>__<split>.*
* ``train_adapter``   -> outputs/<run>/final (adapter), results/training/<run>/*, results/training/curves.png
* ``eval_finetuned``  -> fine-tune predictions/metrics, outputs/merged
* ``awq_benchmark``   -> results/benchmarks/<profile>.json, outputs/awq, benchmark predictions
* ``push_outputs``    -> the same Hub repos/paths the notebooks push to

Steps skip work that is already finished, so an interrupted run can simply be started again. With
``push=True`` (the notebooks) each step uploads its own outputs; the local scripts push once at the end.
Heavy imports (torch, transformers...) happen inside the functions.
"""

from __future__ import annotations

import gc
import json
import os
import random
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from jobx import REPO_ROOT, prompts
from jobx.data.build import read_jsonl, write_jsonl

Log = Callable[[str], None]

BASELINE_FAMILIES: dict[str, dict[str, Any]] = {
    "qwen1.5b": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "prefix": "qwen2.5-1.5b",
        "quant": None,
        "zero_shot": True,
        "big": False,
    },
    "llama1b": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "prefix": "llama3.2-1b",
        "quant": None,
        "zero_shot": False,
        "big": False,
    },
    "qwen7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "prefix": "qwen2.5-7b",
        "quant": "4bit",
        "zero_shot": False,
        "big": True,
    },
}
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class GPUProfile:
    name: str
    dtype: str = "auto"
    train_config: str = "configs/train/qwen2.5-1.5b-qlora.yaml"
    paraphrase: dict[str, Any] = field(default_factory=dict)
    baselines: dict[str, Any] = field(default_factory=dict)
    eval: dict[str, Any] = field(default_factory=dict)
    awq: dict[str, Any] = field(default_factory=dict)


def load_profile(path_or_name: str | Path, root: Path = REPO_ROOT) -> GPUProfile:
    """A profile file, or a name under configs/gpu/ (e.g. 'local_rtx4080')."""
    p = Path(path_or_name)
    if not p.suffix:
        p = root / "configs" / "gpu" / f"{path_or_name}.yaml"
    elif not p.is_absolute() and not p.exists():
        p = root / p
    d = yaml.safe_load(p.read_text())
    prof = GPUProfile(**d)
    for section, keys in {
        "paraphrase": ("n", "batch", "teacher", "nli_model"),
        "baselines": ("batch_small", "batch_7b", "guided_max"),
        "eval": ("batch", "guided_max"),
        "awq": ("n_bench",),
    }.items():
        missing = [k for k in keys if k not in getattr(prof, section)]
        if missing:
            raise ValueError(f"profile {p.name}: {section} is missing {missing}")
    if prof.dtype not in ("auto", "float16", "bfloat16"):
        raise ValueError(f"profile {p.name}: dtype must be auto|float16|bfloat16")
    return prof


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _push(paths: Sequence[Path], root: Path) -> None:
    from jobx.baselines.runs import push_results

    existing = [p.resolve() for p in paths if p.exists()]
    if existing:
        push_results(existing, root)


# --------------------------------------------------------------------------------------------- 01 paraphrase
def paraphrase(
    profile: GPUProfile,
    *,
    push: bool = False,
    do_real_drafts: bool = True,
    root: Path = REPO_ROOT,
    log: Log = print,
) -> dict[str, Any]:
    """Rewrite a train subset with the 7B teacher, filter with rules + NLI, then fetch and draft-label real
    postings. Raw generations are appended per batch, so a restart continues where it stopped."""
    from tqdm.auto import tqdm

    from jobx.baselines.hf_runner import generate, load_model
    from jobx.data.paraphrase import STYLES, filter_paraphrases, paraphrase_messages, pick_subset
    from jobx.data.validate import hf_nli_scorer

    cfg = profile.paraphrase
    out_dir = root / "data" / "paraphrased"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_generations.jsonl"
    train = read_jsonl(root / "data" / "processed" / "train.jsonl")
    subset = pick_subset(train, int(cfg["n"]))
    rng = random.Random(0)
    styles = {e["id"]: rng.choice(STYLES) for e in subset}
    done = {r["id"]: r for r in read_jsonl(raw_path)} if raw_path.exists() else {}
    todo = [e for e in subset if e["id"] not in done]
    log(f"paraphrase: {len(done)} already generated, {len(todo)} to go")
    model = tok = None
    if todo or do_real_drafts:
        model, tok = load_model(cfg["teacher"], quant="4bit", dtype=profile.dtype)
    bs = int(cfg["batch"])
    with open(raw_path, "a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(todo), bs), desc="paraphrase"):
            chunk = todo[i : i + bs]
            msgs = [paraphrase_messages(e["text"], styles[e["id"]]) for e in chunk]
            for e, g in zip(chunk, generate(model, tok, msgs, max_new_tokens=700, sample=True), strict=True):
                row = {"id": e["id"], "style": styles[e["id"]], "output": g.text}
                done[e["id"]] = row
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    nli = hf_nli_scorer(cfg["nli_model"], device=0)
    pairs = [(e, styles[e["id"]], done[e["id"]]["output"]) for e in subset if e["id"] in done]
    kept, stats = filter_paraphrases(pairs, nli=nli, nli_threshold=0.5)
    stats.update({"teacher": cfg["teacher"], "nli_model": cfg["nli_model"], "profile": profile.name})
    write_jsonl(out_dir / "paraphrased_train.jsonl", kept)
    (out_dir / "filter_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    log(f"paraphrase: kept {stats['kept']} of {stats['attempted']} (filter rate {stats['filter_rate']})")
    files = [out_dir / "paraphrased_train.jsonl", out_dir / "filter_stats.json"]
    if do_real_drafts:
        drafts = draft_real_labels(model, tok, train, teacher=cfg["teacher"], root=root, log=log)
        if drafts:
            files += [drafts, root / "data" / "real" / "raw_meta.json"]
    del model
    _free_gpu()
    if push:
        _push(files, root)
    return stats


def draft_real_labels(
    model: Any,
    tok: Any,
    train: Sequence[dict[str, Any]],
    *,
    teacher: str,
    root: Path = REPO_ROOT,
    log: Log = print,
) -> Path | None:
    """Fetch real postings (if not fetched yet) and draft labels with the teacher, 3-shot."""
    from jobx.baselines.hf_runner import generate
    from jobx.schema import JobPosting, parse_output

    raw_path = root / "data" / "real" / "raw_postings.jsonl"
    if not raw_path.exists():
        try:
            from jobx.data.fetch_real import fetch

            log(json.dumps(fetch(root / "configs" / "real_data.yaml", raw_path), indent=2))
        except Exception as e:  # no usable dataset / no network: drafts are optional
            log(f"real postings not fetched: {type(e).__name__}: {e}")
            return None
    out = root / "data" / "real" / "test_real_draft.jsonl"
    if out.exists():
        log(f"drafts already exist: {out}")
        return out
    raw = read_jsonl(raw_path)
    shots = prompts.select_few_shot(train, k=3)
    drafts = []
    for i in range(0, len(raw), 4):
        chunk = raw[i : i + 4]
        gens = generate(model, tok, [prompts.messages(r["text"], shots) for r in chunk], max_new_tokens=400)
        for r, g in zip(chunk, gens, strict=True):
            pr = parse_output(g.text)
            label = pr.posting.to_json_dict() if pr.posting else JobPosting().to_json_dict()
            drafts.append(
                {
                    "id": r["id"],
                    "text": r["text"],
                    "label": label,
                    "meta": {
                        "source": "real",
                        "dataset": r["dataset"],
                        "row": r["row"],
                        "licence": r["licence"],
                        "draft_model": teacher,
                        "draft_schema_ok": pr.schema_ok,
                    },
                }
            )
    write_jsonl(out, drafts)
    log(f"{sum(d['meta']['draft_schema_ok'] for d in drafts)} of {len(drafts)} drafts schema-valid")
    return out


# --------------------------------------------------------------------------------------------- 02 baselines
def run_family(
    family: str,
    profile: GPUProfile,
    *,
    splits: Sequence[str],
    shots: list[dict[str, Any]],
    push: bool = False,
    skip_existing: bool = True,
    root: Path = REPO_ROOT,
) -> list[str]:
    """Zero-/few-shot (free + guided) for one base model; returns the approach ids it produced."""
    from jobx.baselines.hf_runner import load_model
    from jobx.baselines.runs import run_approach

    fam = BASELINE_FAMILIES[family]
    b = profile.baselines
    batch = int(b["batch_7b"] if fam["big"] else b["batch_small"])
    model, tok = load_model(fam["model_id"], quant=fam["quant"], dtype=profile.dtype)
    p, mid = fam["prefix"], fam["model_id"]
    common = {"model_id": mid, "push": push, "skip_existing": skip_existing, "root": root}
    done = []
    for split in splits:
        if fam["zero_shot"]:
            run_approach(
                model,
                tok,
                f"{p}_zero_shot",
                split,
                max_examples=b.get("free_max"),
                batch_size=batch,
                **common,
            )
            run_approach(
                model,
                tok,
                f"{p}_zero_shot_guided",
                split,
                guided=True,
                max_examples=b["guided_max"],
                **common,
            )
            done += [f"{p}_zero_shot", f"{p}_zero_shot_guided"]
        run_approach(
            model,
            tok,
            f"{p}_few_shot",
            split,
            shots=shots,
            max_examples=b.get("free_max"),
            batch_size=max(1, batch // 2),
            **common,
        )
        run_approach(
            model,
            tok,
            f"{p}_few_shot_guided",
            split,
            shots=shots,
            guided=True,
            max_examples=b["guided_max"],
            **common,
        )
        done += [f"{p}_few_shot", f"{p}_few_shot_guided"]
    del model
    _free_gpu()
    return sorted(set(done))


def baselines(
    profile: GPUProfile,
    *,
    families: Sequence[str] = ("qwen1.5b", "qwen7b"),
    push: bool = False,
    root: Path = REPO_ROOT,
    log: Log = print,
) -> list[str]:
    from jobx.baselines.runs import available_test_splits

    splits = available_test_splits(root)
    shots = prompts.select_few_shot(read_jsonl(root / "data" / "processed" / "train.jsonl"), k=3)
    log(f"baselines: splits {splits}, shots {[s['id'] for s in shots]}, families {list(families)}")
    out: list[str] = []
    for fam in families:
        out += run_family(fam, profile, splits=splits, shots=shots, push=push, root=root)
    return out


# --------------------------------------------------------------------------------------------- 03 training
def finalize_training(
    cfg: dict[str, Any], final: Path, *, push: bool = False, root: Path = REPO_ROOT, kind: str = "adapter"
) -> Path:
    """Copy logs to results/training/<run>, plot curves, write the adapter model card; push if asked."""
    from jobx.cards import metrics_table, render_card
    from jobx.eval.curves import plot_curves

    run_dir = root / "results" / "training" / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in ("log_history.json", "best.json", "train_config.json"):
        if (final / f).exists():
            shutil.copy(final / f, run_dir / f)
    png = plot_curves(run_dir / "log_history.json", run_dir / "curves.png")
    shutil.copy(png, root / "results" / "training" / "curves.png")
    (final / "README.md").write_text(adapter_card(cfg, root, metrics_table, render_card))
    if push:
        from jobx.hub import push_folder

        _push(
            [run_dir / f for f in ("log_history.json", "best.json", "train_config.json", "curves.png")]
            + [root / "results" / "training" / "curves.png"],
            root,
        )
        push_folder(final, kind, commit_message="QLoRA adapter")
    return run_dir


def adapter_card(cfg: dict[str, Any], root: Path, metrics_table: Any, render_card: Any) -> str:
    from jobx.hub import repo_id

    def rid(kind: str) -> str:
        try:
            return repo_id(kind)
        except Exception:
            return f"<HF_USERNAME>/{kind}"

    return str(
        render_card(
            "adapter",
            {
                "base_model": cfg["base_model"],
                "repo_name": rid("adapter"),
                "artifacts_repo": rid("artifacts"),
                "n_train": sum(1 for _ in open(root / cfg["data"]["train"], encoding="utf-8")),
                "lora_r": cfg["lora"]["r"],
                "lora_alpha": cfg["lora"]["alpha"],
                "learning_rate": cfg["training"]["learning_rate"],
                "wandb_url": os.environ.get("WANDB_RUN_URL", "see W&B project jobx-qlora"),
                "metrics_table": metrics_table(["qwen2.5-1.5b_qlora"]),
                "base_licence": "Apache-2.0",
            },
        )
    )


def final_dir(cfg: dict[str, Any], root: Path = REPO_ROOT) -> Path:
    return root / cfg["training"]["output_dir"] / "final"


def train_adapter(
    config: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
    push: bool = False,
    force: bool = False,
    root: Path = REPO_ROOT,
    log: Log = print,
) -> Path:
    from jobx.train.config import load_train_config
    from jobx.train.sft import train

    cfg = load_train_config(config, overrides)
    final = final_dir(cfg, root)
    if (final / "best.json").exists() and not force:
        log(f"training already finished: {final} (use --force to retrain)")
    else:
        final = train(cfg)
    finalize_training(cfg, final, push=push, root=root)
    log((final / "best.json").read_text() if (final / "best.json").exists() else "")
    return final


def sweep(*, push: bool = False, root: Path = REPO_ROOT) -> dict[str, Any]:
    from jobx.train.config import sweep_runs
    from jobx.train.sft import train

    summary = {}
    for name, scfg in sweep_runs(root / "configs" / "train" / "sweep.yaml"):
        out = final_dir(scfg, root) if (final_dir(scfg, root) / "best.json").exists() else train(scfg)
        summary[name] = json.loads((out / "best.json").read_text())
    path = root / "results" / "training" / "sweep_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    if push:
        _push([path], root)
    return summary


# --------------------------------------------------------------------------------------------- 04 eval
def resolve_adapter(train_config: str | Path, root: Path = REPO_ROOT) -> str:
    """Local adapter from a finished local run if present, else the Hub repo the notebooks push to."""
    from jobx.train.config import load_train_config

    final = final_dir(load_train_config(train_config), root)
    if (final / "adapter_config.json").exists():
        return str(final)
    from jobx.hub import repo_id

    return repo_id("adapter")


def eval_finetuned(
    profile: GPUProfile,
    *,
    adapter: str | None = None,
    push: bool = False,
    root: Path = REPO_ROOT,
    log: Log = print,
) -> Path:
    from jobx.baselines.hf_runner import load_model
    from jobx.baselines.runs import available_test_splits, run_approach
    from jobx.cards import metrics_table, render_card
    from jobx.train.merge import merge

    adapter = adapter or resolve_adapter(profile.train_config, root)
    log(f"eval: adapter {adapter}")
    e = profile.eval
    model, tok = load_model(BASE_MODEL, adapter=adapter, merge_adapter=True, dtype=profile.dtype)
    for split in available_test_splits(root):
        run_approach(
            model,
            tok,
            "qwen2.5-1.5b_qlora",
            split,
            model_id=adapter,
            batch_size=int(e["batch"]),
            push=push,
            skip_existing=True,
            root=root,
        )
        run_approach(
            model,
            tok,
            "qwen2.5-1.5b_qlora_guided",
            split,
            model_id=adapter,
            guided=True,
            max_examples=e["guided_max"],
            push=push,
            skip_existing=True,
            root=root,
        )
    del model
    _free_gpu()
    out = root / "outputs" / "merged"
    if (out / "config.json").exists():
        log(f"merged model already exists: {out}")
    else:
        out = merge(BASE_MODEL, adapter, out)
    (out / "README.md").write_text(merged_card(adapter, root, metrics_table, render_card))
    if push:
        from jobx.hub import push_folder

        push_folder(out, "merged", commit_message="merged 16-bit weights")
    return out


def merged_card(adapter: str, root: Path, metrics_table: Any, render_card: Any) -> str:
    from jobx.hub import repo_id

    def rid(kind: str) -> str:
        try:
            return repo_id(kind)
        except Exception:
            return f"<HF_USERNAME>/{kind}"

    return str(
        render_card(
            "merged",
            {
                "base_model": BASE_MODEL,
                "adapter_repo": rid("adapter") if Path(adapter).exists() else adapter,
                "repo_name": rid("merged"),
                "artifacts_repo": rid("artifacts"),
                "n_train": sum(
                    1 for _ in open(root / "data" / "processed" / "train.jsonl", encoding="utf-8")
                ),
                "lora_r": 16,
                "lora_alpha": 32,
                "learning_rate": "2e-4",
                "wandb_url": "see W&B project jobx-qlora",
                "metrics_table": metrics_table(["qwen2.5-1.5b_qlora", "qwen2.5-1.5b_qlora_guided"]),
                "base_licence": "Apache-2.0",
            },
        )
    )


# --------------------------------------------------------------------------------------------- 05 AWQ + bench
def resolve_merged(root: Path = REPO_ROOT) -> str:
    local = root / "outputs" / "merged"
    if (local / "config.json").exists():
        return str(local)
    from huggingface_hub import snapshot_download

    from jobx.hub import repo_id

    return str(snapshot_download(repo_id("merged")))


def awq_benchmark(
    profile: GPUProfile, *, push: bool = False, root: Path = REPO_ROOT, log: Log = print
) -> Path:
    """fp16/bf16, bitsandbytes 8-bit and AWQ 4-bit on the same evenly spaced subset, batch size 1."""
    from jobx.baselines.hf_runner import gpu_name, load_model
    from jobx.eval.run import SPLIT_FILES, eval_subset, git_commit
    from jobx.quantize.gpu_bench import awq_quantize, bench_transformers, dir_size_gb

    split = "test_real" if (root / SPLIT_FILES["test_real"]).exists() else "test_synth"
    merged = resolve_merged(root)
    ex = eval_subset(read_jsonl(root / SPLIT_FILES[split]), int(profile.awq["n_bench"]))
    meta = {"hardware": gpu_name(), "commit": git_commit(root), "split": split, "profile": profile.name}
    bench_path = root / "results" / "benchmarks" / f"{profile.name}.json"
    rows: list[dict[str, Any]] = json.loads(bench_path.read_text())["rows"] if bench_path.exists() else []
    have = {r["variant"] for r in rows}
    full_gb = dir_size_gb(merged)

    def bench(variant: str, approach: str, quant: str | None, size: float) -> None:
        if variant in have:
            log(f"{variant}: already benchmarked")
            return
        model, tok = load_model(merged, quant=quant, dtype=profile.dtype)
        rows.append(
            bench_transformers(
                model,
                tok,
                ex,
                variant=variant,
                approach=approach,
                split=split,
                size_gb=size,
                meta={**meta, "model": merged},
                root=root,
                out_pred_dir=root / "results" / "predictions",
            )
        )
        log(json.dumps(rows[-1]))
        del model
        _free_gpu()

    bench("fp16 (transformers)", "qwen2.5-1.5b_merged_fp16_bench", None, full_gb)
    bench("merged 8-bit (bitsandbytes)", "qwen2.5-1.5b_merged_int8", "8bit", round(full_gb / 2, 3))
    awq_dir = root / "outputs" / "awq"
    if "AWQ 4-bit" not in have:
        try:
            if not (awq_dir / "config.json").exists():
                calib = [e["text"] for e in read_jsonl(root / "data" / "processed" / "train.jsonl")[:128]]
                awq_quantize(merged, awq_dir, calib)
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer

            amodel = AutoAWQForCausalLM.from_quantized(str(awq_dir), fuse_layers=False)
            atok = AutoTokenizer.from_pretrained(str(awq_dir))
            atok.padding_side = "left"
            rows.append(
                bench_transformers(
                    amodel.model,
                    atok,
                    ex,
                    variant="AWQ 4-bit",
                    approach="qwen2.5-1.5b_awq",
                    split=split,
                    size_gb=dir_size_gb(awq_dir),
                    meta={**meta, "model": "awq"},
                    root=root,
                    out_pred_dir=root / "results" / "predictions",
                )
            )
            log(json.dumps(rows[-1]))
            if push:
                from jobx.hub import push_folder

                push_folder(awq_dir, "awq", commit_message="AWQ 4-bit")
        except Exception as e:  # AutoAWQ is archived upstream; keep the other rows
            log(f"AWQ step failed, row stays pending: {e!r}")
    bench_path.parent.mkdir(parents=True, exist_ok=True)
    bench_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    if push:
        files = [bench_path]
        for a in ("qwen2.5-1.5b_merged_int8", "qwen2.5-1.5b_awq", "qwen2.5-1.5b_merged_fp16_bench"):
            files += [root / "results" / "predictions" / f"{a}__{split}{s}" for s in (".jsonl", ".meta.json")]
        _push(files, root)
    return bench_path


# --------------------------------------------------------------------------------------------- push
def collect_push_files(root: Path = REPO_ROOT) -> list[Path]:
    """Every result file the notebooks push to the artifacts dataset repo (same relative paths)."""
    pats = [
        "results/predictions/*",
        "results/metrics/*",
        "results/benchmarks/*.json",
        "results/training/**/*",
        "data/paraphrased/paraphrased_train.jsonl",
        "data/paraphrased/filter_stats.json",
        "data/real/test_real_draft.jsonl",
        "data/real/raw_meta.json",
    ]
    files: list[Path] = []
    for pat in pats:
        files += [p for p in root.glob(pat) if p.is_file() and p.name != "README.md"]
    return sorted(set(files))


def push_outputs(
    profile: GPUProfile, *, root: Path = REPO_ROOT, log: Log = print, dry_run: bool = False
) -> dict[str, Any]:
    """Upload results to <user>/jobx-artifacts and the adapter / merged / AWQ folders to their model repos."""
    from jobx.train.config import load_train_config

    files = collect_push_files(root)
    final = final_dir(load_train_config(profile.train_config), root)
    folders = {"adapter": final, "merged": root / "outputs" / "merged", "awq": root / "outputs" / "awq"}
    folders = {
        k: v
        for k, v in folders.items()
        if (v / "config.json").exists() or (v / "adapter_config.json").exists()
    }
    plan = {
        "artifact_files": [str(p.relative_to(root)) for p in files],
        "model_folders": {k: str(v.relative_to(root)) for k, v in folders.items()},
    }
    if dry_run:
        return plan
    from jobx.baselines.runs import push_results
    from jobx.hub import push_folder

    if files:
        log(f"artifacts: {push_results(files, root)} ({len(files)} files)")
    for kind, path in folders.items():
        log(f"{kind}: {push_folder(path, kind, commit_message=f'{kind} from local GPU ({profile.name})')}")
    return plan
