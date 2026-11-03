"""QLoRA supervised fine-tuning with TRL ``SFTTrainer`` (runs on a Colab T4).

* 4-bit NF4 base weights with double quantisation, fp16 compute (the T4 has no bf16);
* LoRA on all linear projections, gradient checkpointing, paged 8-bit AdamW;
* completion-only loss: the data is conversational prompt/completion, so only the assistant
  JSON tokens contribute to the loss;
* validation JSON micro F1 (greedy decoding on a val subset) is computed at every evaluation
  and drives checkpoint selection and early stopping;
* Weights & Biases logs loss, learning rate, eval metrics and sample generations.

Usage (from the repo root)::

    python -m jobx.train.sft --config configs/train/qwen2.5-1.5b-qlora.yaml [--set training.learning_rate=1e-4]
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from jobx import prompts
from jobx.data.build import read_jsonl
from jobx.eval.metrics import aggregate, score_example
from jobx.train.config import load_train_config, parse_overrides


def build_datasets(cfg: dict[str, Any]) -> tuple[Any, Any, list[dict[str, Any]]]:
    from datasets import Dataset

    d = cfg["data"]
    train = read_jsonl(d["train"])
    para = d.get("paraphrased")
    if para and Path(para).exists():
        train += read_jsonl(para)
    val = read_jsonl(d["val"])
    rng = random.Random(cfg["training"].get("seed", 42))
    rng.shuffle(train)
    train_ds = Dataset.from_list([prompts.sft_record(e) for e in train])
    val_ds = Dataset.from_list([prompts.sft_record(e) for e in val])
    n_sub = int(d.get("val_subset_for_json_f1", 64))
    val_sub = sorted(val, key=lambda e: e["id"])[:: max(1, len(val) // n_sub)][:n_sub]
    return train_ds, val_ds, val_sub


def make_json_f1_callback(
    val_examples: list[dict[str, Any]], tokenizer: Any, n_log: int, use_wandb: bool, system: str = "full"
) -> Any:
    import torch
    from transformers import TrainerCallback

    class JsonF1Callback(TrainerCallback):
        """Adds ``eval_json_f1`` to the eval metrics (in place, before early stopping reads them)."""

        def on_evaluate(
            self,
            args: Any,
            state: Any,
            control: Any,
            metrics: dict[str, float] | None = None,
            model: Any = None,
            **kw: Any,
        ) -> None:
            if model is None or metrics is None:
                return
            was_training = model.training
            model.eval()
            use_cache = getattr(model.config, "use_cache", False)
            model.config.use_cache = True
            tokenizer.padding_side = "left"
            outs: list[str] = []
            bs = 8
            for i in range(0, len(val_examples), bs):
                chunk = val_examples[i : i + bs]
                texts = [
                    tokenizer.apply_chat_template(
                        prompts.messages(e["text"], system=system), tokenize=False, add_generation_prompt=True
                    )
                    for e in chunk
                ]
                enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(
                    model.device
                )
                with torch.inference_mode():
                    gen = model.generate(
                        **enc, max_new_tokens=320, do_sample=False, pad_token_id=tokenizer.pad_token_id
                    )
                outs += tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            model.config.use_cache = use_cache
            tokenizer.padding_side = "right"
            if was_training:
                model.train()
            scores = [score_example(e["id"], e["label"], o) for e, o in zip(val_examples, outs, strict=True)]
            agg = aggregate(scores, n_boot=0)
            metrics["eval_json_f1"] = float(agg["micro_f1"])
            metrics["eval_json_validity"] = float(agg["json_validity"])
            metrics["eval_hallucination_rate"] = float(agg["hallucination_rate"])
            state.log_history.append(
                {
                    "step": state.global_step,
                    "eval_json_f1": metrics["eval_json_f1"],
                    "eval_json_validity": metrics["eval_json_validity"],
                    "eval_hallucination_rate": metrics["eval_hallucination_rate"],
                }
            )
            if use_wandb:
                import wandb

                if wandb.run is not None:
                    table = wandb.Table(columns=["id", "posting", "gold", "prediction"])
                    for e, o in list(zip(val_examples, outs, strict=True))[:n_log]:
                        table.add_data(e["id"], e["text"][:600], prompts.target_json(e["label"]), o)
                    wandb.log(
                        {
                            "eval/json_f1": metrics["eval_json_f1"],
                            "eval/json_validity": metrics["eval_json_validity"],
                            "eval/hallucination_rate": metrics["eval_hallucination_rate"],
                            f"eval/samples_step_{state.global_step}": table,
                            "train/global_step": state.global_step,
                        }
                    )

    return JsonF1Callback()


def train(cfg: dict[str, Any]) -> Path:
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer

    t = cfg["training"]
    q = cfg["quantization"]
    lc = cfg["lora"]
    use_wandb = t.get("report_to") == "wandb" and bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", cfg.get("wandb", {}).get("project", "jobx-qlora"))
        os.environ.setdefault("WANDB_LOG_MODEL", "false")

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb,
        torch_dtype=torch.bfloat16 if t.get("bf16") else torch.float16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    peft_cfg = LoraConfig(
        r=lc["r"],
        lora_alpha=lc["alpha"],
        lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"],
        bias=lc["bias"],
        task_type="CAUSAL_LM",
    )
    train_ds, val_ds, val_sub = build_datasets(cfg)
    args = SFTConfig(
        output_dir=t["output_dir"],
        run_name=cfg["run_name"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        max_length=t["max_length"],
        optim=t["optim"],
        fp16=t["fp16"],
        bf16=t["bf16"],
        tf32=t.get("tf32"),
        dataloader_num_workers=int(t.get("dataloader_num_workers", 0)),
        gradient_checkpointing=t["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        greater_is_better=t["greater_is_better"],
        seed=t["seed"],
        report_to=["wandb"] if use_wandb else [],
        completion_only_loss=True,  # loss on the assistant JSON only
        packing=False,
        dataset_num_proc=2,
    )
    callbacks = [
        make_json_f1_callback(
            val_sub, tok, int(cfg.get("wandb", {}).get("log_sample_generations", 8)), use_wandb
        ),
        EarlyStoppingCallback(early_stopping_patience=t["early_stopping_patience"]),
    ]
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        peft_config=peft_cfg,
        callbacks=callbacks,
    )
    trainer.model.print_trainable_parameters()
    resume = bool(t.get("resume", True)) and any(Path(t["output_dir"]).glob("checkpoint-*"))
    if resume:
        print(f"resuming from the last checkpoint in {t['output_dir']}")
    trainer.train(resume_from_checkpoint=True if resume else None)
    final = Path(t["output_dir"]) / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    (final / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    (final / "train_config.json").write_text(json.dumps(cfg, indent=2))
    best = {
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "global_step": trainer.state.global_step,
    }
    (final / "best.json").write_text(json.dumps(best, indent=2))
    if use_wandb:
        import wandb

        wandb.finish()
    return final


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], help="dotted override, e.g. lora.r=8")
    a = ap.parse_args(argv)
    cfg = load_train_config(a.config, parse_overrides(a.set))
    out = train(cfg)
    print(f"adapter saved to {out}")


if __name__ == "__main__":
    main()
