"""Merge a LoRA adapter into its fp16 base model and (optionally) push it to the Hub.

    python -m jobx.train.merge --base Qwen/Qwen2.5-1.5B-Instruct --adapter <user>/jobx-qwen2.5-1.5b-qlora \
        --out outputs/merged [--push]

Merging is done in fp16 on whatever device is available (a 1.5B model needs ~3.5 GB, so a
Colab CPU runtime also works). The merged checkpoint is what llama.cpp converts to GGUF and
what AutoAWQ quantises.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def merge(base: str, adapter: str, out: str | Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map=device_map)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size="2GB")
    tok = AutoTokenizer.from_pretrained(adapter if Path(adapter).exists() else base)
    tok.save_pretrained(out)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True, help="local adapter dir or Hub repo id")
    ap.add_argument("--out", default="outputs/merged")
    ap.add_argument("--push", action="store_true", help="push to <user>/<repos.merged> from configs/hub.yaml")
    ap.add_argument("--card", default=None, help="README.md (model card) to include")
    a = ap.parse_args(argv)
    out = merge(a.base, a.adapter, a.out)
    if a.card:
        (out / "README.md").write_text(Path(a.card).read_text())
    if a.push:
        from jobx.hub import push_folder

        rid = push_folder(out, "merged", commit_message="merged fp16 weights")
        print(f"pushed to https://huggingface.co/{rid}")
    print(f"merged model saved to {out}")


if __name__ == "__main__":
    main()
