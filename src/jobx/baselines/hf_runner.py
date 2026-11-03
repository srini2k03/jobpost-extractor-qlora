"""GPU inference helpers (transformers) shared by the Colab notebooks.

Heavy imports are lazy so the package stays importable on CPU-only CI.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jobx import prompts
from jobx.schema import json_schema


@dataclass
class GenOutput:
    text: str
    latency_s: float
    new_tokens: int
    hit_max_tokens: bool


def resolve_dtype(dtype: str = "auto") -> Any:
    """'auto' -> torch.bfloat16 on GPUs that support it (RTX 30xx/40xx, A100...), else torch.float16."""
    import torch

    if dtype == "auto":
        return (
            torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        )
    return getattr(torch, dtype)


def gpu_name() -> str:
    try:
        import torch

        return str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU"
    except Exception:
        return "unknown"


def load_model(
    model_id: str,
    *,
    quant: str | None = None,  # None (16-bit) | "4bit" | "8bit"
    adapter: str | None = None,
    merge_adapter: bool = False,
    dtype: str = "auto",  # "auto" = bfloat16 where supported (Ampere/Ada), else float16 (T4)
) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tdtype = resolve_dtype(dtype)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs: dict[str, Any] = {"torch_dtype": tdtype, "device_map": "auto"}
    if quant == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=tdtype,
        )
    elif quant == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        if merge_adapter:
            model = model.merge_and_unload()
    model.eval()
    return model, tok


def _prompt_text(tok: Any, msgs: list[dict[str, str]]) -> str:
    text: str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text


def generate(
    model: Any,
    tok: Any,
    batch_msgs: Sequence[list[dict[str, str]]],
    *,
    max_new_tokens: int = 384,
    prefix_allowed_tokens_fn: Callable[..., Any] | None = None,
    sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[GenOutput]:
    """Greedy by default (all evaluation runs); ``sample=True`` is only used for paraphrasing."""
    import torch

    texts = [_prompt_text(tok, m) for m in batch_msgs]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=temperature if sample else None,
            top_p=top_p if sample else None,
            top_k=None,
            pad_token_id=tok.pad_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )
    dt = time.perf_counter() - t0
    res = []
    plen = enc["input_ids"].shape[1]
    for row in out:
        new = row[plen:]
        n_new = int((new != tok.pad_token_id).sum())
        res.append(
            GenOutput(
                tok.decode(new, skip_special_tokens=True), dt / len(texts), n_new, n_new >= max_new_tokens
            )
        )
    return res


def guided_prefix_fn(tok: Any) -> Callable[..., Any]:
    """lm-format-enforcer constraint: the output must be JSON matching the JobPosting schema."""
    from jobx.baselines.guided import build_transformers_prefix_fn

    return build_transformers_prefix_fn(tok, json_schema())


def run_predictions(
    model: Any,
    tok: Any,
    examples: Sequence[dict[str, Any]],
    out_path: str | Path,
    *,
    meta: dict[str, Any],
    shots: Sequence[dict[str, Any]] = (),
    guided: bool = False,
    batch_size: int = 8,
    max_new_tokens: int = 384,
    progress: bool = True,
) -> Path:
    """Write ``<approach>__<split>.jsonl`` + ``.meta.json``. Guided decoding runs with batch size 1."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    prefix_fn = guided_prefix_fn(tok) if guided else None
    bs = 1 if guided else batch_size
    it: Any = range(0, len(examples), bs)
    if progress:
        try:
            from tqdm.auto import tqdm

            it = tqdm(it, desc=out.stem)
        except ImportError:
            pass
    total_new, total_time = 0, 0.0
    with open(out, "w", encoding="utf-8") as f:
        for i in it:
            chunk = examples[i : i + bs]
            msgs = [prompts.messages(e["text"], shots) for e in chunk]
            gens = generate(
                model, tok, msgs, max_new_tokens=max_new_tokens, prefix_allowed_tokens_fn=prefix_fn
            )
            for e, g in zip(chunk, gens, strict=True):
                total_new += g.new_tokens
                total_time += g.latency_s
                f.write(
                    json.dumps(
                        {
                            "id": e["id"],
                            "output": g.text,
                            "latency_s": round(g.latency_s, 4),
                            "new_tokens": g.new_tokens,
                            "hit_max_tokens": g.hit_max_tokens,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    meta = {
        **meta,
        "decoding": "guided" if guided else "free",
        "n_shots": len(shots),
        "shot_ids": [s["id"] for s in shots],
        "batch_size": bs,
        "max_new_tokens": max_new_tokens,
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "throughput_tokens_per_s": round(total_new / total_time, 2) if total_time else None,
    }
    Path(str(out).removesuffix(".jsonl") + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out
