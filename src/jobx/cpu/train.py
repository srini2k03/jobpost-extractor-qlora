"""LoRA fine-tune of Qwen2.5-0.5B-Instruct on CPU (GitHub-hosted runners), resumable across jobs.

    python -m jobx.cpu.train bench      # token-length stats, max_length, seconds/step -> plan.json
    python -m jobx.cpu.train train      # train until done or the per-job time limit; checkpoint; status.json
    python -m jobx.cpu.train status     # exit 0 when training is complete, 3 otherwise

State lives in ``outputs/cpu-lora`` (plan.json, status.json, checkpoint-*, final/). The workflow tars it
into a GitHub Release asset between jobs, so the next job resumes from the last checkpoint. Curves and
logs are written to ``results/training/<run_name>/`` after every job (this replaces W&B).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from jobx import REPO_ROOT, prompts
from jobx.cpu.plan import choose_max_length, length_stats, plan_training, stratified_subset
from jobx.data.build import read_jsonl

EXIT_DONE, EXIT_NOT_DONE = 0, 3


def load_route_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else REPO_ROOT / "configs" / "cpu" / "route.yaml"
    cfg: dict[str, Any] = yaml.safe_load(p.read_text())
    return cfg


def state_dir(cfg: dict[str, Any], root: Path = REPO_ROOT) -> Path:
    return root / cfg["output_dir"]


def results_dir(cfg: dict[str, Any], root: Path = REPO_ROOT) -> Path:
    return root / "results" / "training" / cfg["run_name"]


def cpu_supports_bf16() -> bool:
    """bf16 matmuls are only fast with AVX512-BF16 or AMX; otherwise fp32 is faster on CPU."""
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    return "avx512_bf16" in flags or "amx_bf16" in flags


def set_threads() -> int:
    import torch

    n = os.cpu_count() or 4
    torch.set_num_threads(n)
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    return n


def token_lengths(examples: list[dict[str, Any]], tok: Any, system: str) -> list[int]:
    out = []
    for e in examples:
        r = prompts.sft_record(e, system=system)
        ids = tok.apply_chat_template(r["prompt"] + r["completion"], tokenize=True)
        out.append(len(ids))
    return out


def _sft_config(
    cfg: dict[str, Any],
    out_dir: Path,
    *,
    max_steps: int,
    max_length: int,
    eval_steps: int | None,
    bf16: bool,
    save: bool = True,
) -> Any:
    from trl import SFTConfig

    t = cfg["training"]
    return SFTConfig(
        output_dir=str(out_dir),
        run_name=cfg["run_name"],
        max_steps=max_steps,
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        per_device_eval_batch_size=t["per_device_train_batch_size"],
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        max_length=max_length,
        use_cpu=True,
        bf16=bf16,
        fp16=False,
        gradient_checkpointing=t.get("gradient_checkpointing", False),
        logging_steps=t["logging_steps"],
        eval_strategy="steps" if eval_steps else "no",
        eval_steps=eval_steps,
        save_strategy="steps" if save else "no",
        save_steps=t["save_every_steps"],
        save_total_limit=2,
        report_to=[],
        seed=t["seed"],
        completion_only_loss=True,
        packing=False,
        dataset_num_proc=1,
        dataloader_num_workers=0,
    )


def _model_and_tok(cfg: dict[str, Any], bf16: bool) -> tuple[Any, Any, Any]:
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16 if bf16 else torch.float32
    )
    model.config.use_cache = False
    lc = cfg["lora"]
    peft_cfg = LoraConfig(
        r=lc["r"],
        lora_alpha=lc["alpha"],
        lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return model, tok, peft_cfg


def _usable_examples(
    cfg: dict[str, Any], tok: Any, max_length: int, root: Path
) -> tuple[list[dict[str, Any]], int]:
    train = read_jsonl(root / cfg["data"]["train"])
    lens = token_lengths(train, tok, cfg["system_prompt"])
    keep = [e for e, n in zip(train, lens, strict=True) if n <= max_length]
    return keep, len(train) - len(keep)


def bench(cfg: dict[str, Any], root: Path = REPO_ROOT) -> dict[str, Any]:
    """Measure token lengths and seconds per optimiser step on this machine; write plan.json."""
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTTrainer

    threads = set_threads()
    bf16 = cpu_supports_bf16()
    model, tok, peft_cfg = _model_and_tok(cfg, bf16)
    t = cfg["training"]
    train = read_jsonl(root / cfg["data"]["train"])
    lens = token_lengths(train, tok, cfg["system_prompt"])
    max_length = choose_max_length(lens, target=t["max_length_target"], cap=t["max_length_cap"])
    usable = [e for e, n in zip(train, lens, strict=True) if n <= max_length]
    eff = t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]
    n_steps = int(cfg["budget"]["bench_steps"])
    sample = random.Random(0).sample(usable, min(len(usable), n_steps * eff))
    ds = Dataset.from_list([prompts.sft_record(e, system=cfg["system_prompt"]) for e in sample])
    stamps: list[float] = []

    class Timer(TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
            stamps.append(time.perf_counter())

    out = state_dir(cfg, root) / "bench"
    args = _sft_config(
        cfg, out, max_steps=n_steps, max_length=max_length, eval_steps=None, bf16=bf16, save=False
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
        callbacks=[Timer()],
    )
    t0 = time.perf_counter()
    trainer.train()
    steps = [b - a for a, b in zip([t0, *stamps[:-1]], stamps, strict=True)]
    per_step = sum(steps[1:]) / max(1, len(steps) - 1) if len(steps) > 1 else steps[0]
    b = cfg["budget"]
    plan = plan_training(
        seconds_per_step=per_step,
        effective_batch=eff,
        n_available=len(usable),
        max_length=max_length,
        dropped_too_long=len(train) - len(usable),
        job_minutes=b["job_minutes"],
        max_jobs=b["max_jobs"],
        overhead=b["overhead"],
        max_epochs=t["max_epochs"],
    )
    info = {
        "plan": plan.as_dict(),
        "token_length": length_stats(lens),
        "step_seconds": [round(s, 2) for s in steps],
        "threads": threads,
        "bf16": bf16,
        "cpu": _cpu_name(),
        "base_model": cfg["base_model"],
        "system_prompt": cfg["system_prompt"],
    }
    sd = state_dir(cfg, root)
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "plan.json").write_text(json.dumps(info, indent=2) + "\n")
    rd = results_dir(cfg, root)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "plan.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def latest_checkpoint(out_dir: Path) -> Path | None:
    cks = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return cks[-1] if cks else None


def make_time_limit_callback(max_minutes: float) -> Any:
    from transformers import TrainerCallback

    class TimeLimit(TrainerCallback):
        """Save and stop cleanly before the GitHub job limit; the next job resumes from the checkpoint."""

        def __init__(self) -> None:
            self.start = time.time()
            self.stopped = False

        def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
            if (time.time() - self.start) / 60 > max_minutes and state.global_step < state.max_steps:
                control.should_save = True
                control.should_training_stop = True
                self.stopped = True

    return TimeLimit()


def train(cfg: dict[str, Any], *, max_minutes: float | None = None, root: Path = REPO_ROOT) -> dict[str, Any]:
    from datasets import Dataset
    from trl import SFTTrainer

    from jobx.eval.curves import plot_curves
    from jobx.train.sft import make_json_f1_callback

    sd = state_dir(cfg, root)
    if not (sd / "plan.json").exists():
        bench(cfg, root)
    info = json.loads((sd / "plan.json").read_text())
    plan = info["plan"]
    set_threads()
    bf16 = bool(info.get("bf16"))
    model, tok, peft_cfg = _model_and_tok(cfg, bf16)
    usable, _ = _usable_examples(cfg, tok, plan["max_length"], root)
    subset = stratified_subset(usable, plan["n_train"])
    random.Random(cfg["training"]["seed"]).shuffle(subset)
    system = cfg["system_prompt"]
    train_ds = Dataset.from_list([prompts.sft_record(e, system=system) for e in subset])
    val = sorted(read_jsonl(root / cfg["data"]["val"]), key=lambda e: e["id"])
    d = cfg["data"]
    val_loss = Dataset.from_list(
        [
            prompts.sft_record(e, system=system)
            for e in val[:: max(1, len(val) // d["val_loss_subset"])][: d["val_loss_subset"]]
        ]
    )
    val_f1 = val[:: max(1, len(val) // d["val_json_f1_subset"])][: d["val_json_f1_subset"]]
    eval_steps = max(1, math.ceil(plan["max_steps"] / cfg["training"]["evals_per_run"]))
    args = _sft_config(
        cfg, sd, max_steps=plan["max_steps"], max_length=plan["max_length"], eval_steps=eval_steps, bf16=bf16
    )
    limit = make_time_limit_callback(max_minutes if max_minutes is not None else cfg["budget"]["job_minutes"])
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_loss,
        processing_class=tok,
        peft_config=peft_cfg,
        callbacks=[make_json_f1_callback(val_f1, tok, 0, False, system=system), limit],
    )
    ck = latest_checkpoint(sd)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=str(ck) if ck else None)
    done = trainer.state.global_step >= plan["max_steps"]
    if done:
        final = sd / "final"
        trainer.save_model(str(final))
        tok.save_pretrained(str(final))
    status = {
        "done": done,
        "global_step": trainer.state.global_step,
        "max_steps": plan["max_steps"],
        "resumed_from": ck.name if ck else None,
        "job_minutes": round((time.time() - t0) / 60, 1),
        "stopped_by_time_limit": limit.stopped,
    }
    (sd / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    rd = results_dir(cfg, root)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2) + "\n")
    (rd / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    plot_curves(rd / "log_history.json", rd / "curves.png")
    if done:
        import shutil

        shutil.copy(rd / "curves.png", root / "results" / "training" / "curves.png")
    return status


def status(cfg: dict[str, Any], root: Path = REPO_ROOT) -> dict[str, Any]:
    p = state_dir(cfg, root) / "status.json"
    st: dict[str, Any] = json.loads(p.read_text()) if p.exists() else {"done": False, "global_step": 0}
    st["final_adapter"] = (state_dir(cfg, root) / "final" / "adapter_config.json").exists()
    return st


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["bench", "train", "status"])
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--max-minutes", type=float, default=None, help="train: stop and checkpoint after this long"
    )
    ap.add_argument("--root", default=str(REPO_ROOT), help="repository root for outputs/ and results/")
    a = ap.parse_args(argv)
    cfg = load_route_config(a.config)
    root = Path(a.root)
    if a.command == "bench":
        print(json.dumps(bench(cfg, root), indent=2))
        return 0
    if a.command == "train":
        st = train(cfg, max_minutes=a.max_minutes, root=root)
        print(json.dumps(st, indent=2))
        return EXIT_DONE if st["done"] else EXIT_NOT_DONE
    st = status(cfg, root)
    print(json.dumps(st, indent=2))
    return EXIT_DONE if st["done"] and st["final_adapter"] else EXIT_NOT_DONE


if __name__ == "__main__":
    sys.exit(main())
