"""Compute-aware LoRA study: one reduced-budget run per configuration, selected on validation only.

    python -m jobx.cpu.study train --config-id r16_all       # train one config of configs/lora_study.yaml
    python -m jobx.cpu.study score --config-id r16_all       # score its val predictions -> results/lora_study/<id>.json
    python -m jobx.cpu.study summary                          # results/lora_study/summary.json (+ selection)

Every configuration gets the SAME budget (a stratified subset of the training split, a fixed number of
optimiser steps, the same seed, prompt and maximum length), so differences come from the configuration.
This is deliberately smaller than the production run (2,177 examples, 2 epochs); results are labelled as a
reduced-budget sweep. Validation predictions are made with llama.cpp on the merged Q8_0 GGUF, exactly like
every other evaluation in the repo; the test splits are never used here.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import yaml

from jobx import REPO_ROOT, prompts
from jobx.data.build import read_jsonl

STUDY = REPO_ROOT / "configs" / "lora_study.yaml"
OUT = REPO_ROOT / "results" / "lora_study"


def load_study(path: Path = STUDY) -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load(path.read_text())
    return cfg


def resolve(study: dict[str, Any], config_id: str) -> dict[str, Any]:
    """Shared settings merged with one configuration's overrides."""
    base = dict(study["shared"])
    for c in study["configs"]:
        if c["id"] == config_id:
            return {**base, **c}
    raise KeyError(config_id)


def train_one(study: dict[str, Any], config_id: str, out_dir: Path, root: Path = REPO_ROOT) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from jobx.cpu.plan import stratified_subset
    from jobx.cpu.train import cpu_supports_bf16, set_threads

    c = resolve(study, config_id)
    threads = set_threads()
    bf16 = cpu_supports_bf16()
    tok = AutoTokenizer.from_pretrained(c["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        c["base_model"], torch_dtype=torch.bfloat16 if bf16 else torch.float32
    )
    model.config.use_cache = False
    total_params = sum(p.numel() for p in model.parameters())
    train = read_jsonl(root / "data/processed/train.jsonl")
    subset = stratified_subset(train, int(c["n_train"]))
    ds = Dataset.from_list([prompts.sft_record(e, system=c["system_prompt"]) for e in subset])
    val = sorted(read_jsonl(root / "data/processed/val.jsonl"), key=lambda e: e["id"])
    k = int(c["val_loss_n"])
    val_loss = Dataset.from_list(
        [prompts.sft_record(e, system=c["system_prompt"]) for e in val[:: max(1, len(val) // k)][:k]]
    )
    peft_cfg = LoraConfig(
        r=int(c["r"]),
        lora_alpha=int(c["alpha"]),
        lora_dropout=float(c["dropout"]),
        target_modules=list(c["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir=str(out_dir / "trainer"),
        max_steps=int(c["max_steps"]),
        per_device_train_batch_size=int(c["batch_size"]),
        gradient_accumulation_steps=int(c["grad_accum"]),
        per_device_eval_batch_size=int(c["batch_size"]),
        learning_rate=float(c["learning_rate"]),
        lr_scheduler_type=c["lr_scheduler_type"],
        warmup_ratio=float(c["warmup_ratio"]),
        weight_decay=float(c["weight_decay"]),
        max_grad_norm=1.0,
        max_length=int(c["max_length"]),
        use_cpu=True,
        bf16=bf16,
        fp16=False,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=int(c["max_steps"]),
        save_strategy="no",
        report_to=[],
        seed=int(c["seed"]),
        completion_only_loss=True,
        packing=False,
        dataset_num_proc=1,
        dataloader_num_workers=0,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        eval_dataset=val_loss,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    t0 = time.perf_counter()
    trainer.train()
    train_s = time.perf_counter() - t0
    adapter = out_dir / "adapter"
    trainer.save_model(str(adapter))
    tok.save_pretrained(str(adapter))
    adapter_mb = sum(f.stat().st_size for f in adapter.glob("*.safetensors")) / 1e6
    logs = trainer.state.log_history
    train_losses = [x["loss"] for x in logs if "loss" in x]
    eval_losses = [x["eval_loss"] for x in logs if "eval_loss" in x]
    info = {
        "id": config_id,
        "config": {
            k2: c[k2]
            for k2 in (
                "r",
                "alpha",
                "dropout",
                "learning_rate",
                "target_modules",
                "max_length",
                "warmup_ratio",
                "weight_decay",
                "batch_size",
                "grad_accum",
            )
        },
        "budget": {
            "n_train": len(subset),
            "max_steps": int(c["max_steps"]),
            "effective_batch": int(c["batch_size"]) * int(c["grad_accum"]),
        },
        "params_total": total_params,
        "params_trainable": trainable,
        "trainable_share": round(trainable / total_params, 6),
        "adapter_mb": round(adapter_mb, 2),
        "train_seconds": round(train_s, 1),
        "seconds_per_step": round(train_s / int(c["max_steps"]), 2),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2),
        "final_train_loss": round(float(train_losses[-1]), 4) if train_losses else None,
        "val_loss": round(float(eval_losses[-1]), 4) if eval_losses else None,
        "loss_curve": [round(float(v), 4) for v in train_losses],
        "threads": threads,
        "bf16": bf16,
        "cpu": _cpu(),
    }
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def _cpu() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def score_one(config_id: str, work: Path, root: Path = REPO_ROOT, out: Path = OUT) -> dict[str, Any]:
    from jobx.eval.metrics import aggregate, score_example

    info = json.loads((work / "train_info.json").read_text())
    gold = {g["id"]: g for g in read_jsonl(root / "data/processed/val.jsonl")}
    preds = read_jsonl(work / "pred" / f"study-{config_id}__val.jsonl")
    scores = [score_example(p["id"], gold[p["id"]]["label"], p.get("output") or "") for p in preds]
    m = aggregate(scores, n_boot=1000)
    info["val"] = {
        "n": m["n"],
        **{
            k: round(float(m[k]), 4)
            for k in (
                "micro_f1",
                "macro_f1",
                "json_validity",
                "schema_validity",
                "hallucination_rate",
                "field_em_mean",
            )
        },
        "ci95_micro_f1": [round(float(x), 4) for x in m["ci95"]["micro_f1"]],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{config_id}.json").write_text(json.dumps(info, indent=2) + "\n")
    (out / "predictions").mkdir(exist_ok=True)
    (out / "predictions" / f"{config_id}__val.jsonl").write_text(
        "".join(json.dumps({"id": p["id"], "output": p["output"]}, ensure_ascii=False) + "\n" for p in preds)
    )
    return info


def summary(out: Path = OUT, study: dict[str, Any] | None = None) -> dict[str, Any]:
    study = study or load_study()
    runs = [json.loads(p.read_text()) for p in sorted(out.glob("*.json")) if p.name != "summary.json"]
    runs = [r for r in runs if "val" in r]
    if not runs:
        return {}
    best = max(runs, key=lambda r: (r["val"]["micro_f1"], -r["params_trainable"]))
    lo, _ = best["val"]["ci95_micro_f1"]
    ties = [r["id"] for r in runs if r["id"] != best["id"] and r["val"]["micro_f1"] >= lo]
    res = {
        "budget_note": study.get("note"),
        "selection": "highest validation micro-F1 (fewer trainable parameters breaks ties); test splits unused",
        "best": best["id"],
        "within_best_ci": ties,
        "runs": [
            {
                k: r[k]
                for k in (
                    "id",
                    "config",
                    "params_trainable",
                    "trainable_share",
                    "adapter_mb",
                    "train_seconds",
                    "seconds_per_step",
                    "peak_rss_gb",
                    "final_train_loss",
                    "val_loss",
                    "val",
                    "cpu",
                    "bf16",
                )
            }
            for r in sorted(runs, key=lambda r: -r["val"]["micro_f1"])
        ],
    }
    (out / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


def matrix(study: dict[str, Any]) -> list[dict[str, str]]:
    return [{"id": c["id"]} for c in study["configs"]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["train", "score", "summary", "matrix"])
    ap.add_argument("--config-id", default=None)
    ap.add_argument("--work", default="outputs/study")
    a = ap.parse_args(argv)
    study = load_study()
    if a.command == "matrix":
        print(json.dumps({"include": matrix(study)}))
        return 0
    if a.command == "summary":
        print(json.dumps(summary(OUT, study), indent=2))
        return 0
    if not a.config_id:
        ap.error("--config-id is required")
    work = Path(a.work) / a.config_id
    work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if a.command == "train":
        print(json.dumps(train_one(study, a.config_id, work), indent=2))
    else:
        print(json.dumps(score_one(a.config_id, work), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
