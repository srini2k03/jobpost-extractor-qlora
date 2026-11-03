"""T4 benchmark helpers for notebook 05 (fp16 / bitsandbytes 8-bit / AWQ 4-bit)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jobx.baselines.hf_runner import run_predictions
from jobx.eval.metrics import aggregate
from jobx.eval.run import score_file


def dir_size_gb(path: str | Path) -> float:
    p = Path(path)
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.suffix in (".safetensors", ".bin", ".gguf")]
    return round(sum(f.stat().st_size for f in files) / 1e9, 3)


def bench_transformers(
    model: Any,
    tok: Any,
    examples: Sequence[dict[str, Any]],
    *,
    variant: str,
    approach: str,
    split: str,
    size_gb: float | None,
    meta: dict[str, Any],
    out_pred_dir: str | Path = "results/predictions",
    root: str | Path = ".",
    batch_size: int = 1,
) -> dict[str, Any]:
    """Batch size 1 by default so latency is per posting (the serving case)."""
    import torch

    torch.cuda.reset_peak_memory_stats()
    pred = run_predictions(
        model,
        tok,
        examples,
        Path(out_pred_dir) / f"{approach}__{split}.jsonl",
        meta={**meta, "approach": approach},
        batch_size=batch_size,
    )
    peak = torch.cuda.max_memory_allocated() / 1e9
    rows = [json.loads(x) for x in pred.read_text().splitlines()]
    lat = [r["latency_s"] for r in rows]
    toks = sum(r["new_tokens"] for r in rows)
    scores, _ = score_file(pred, root)
    agg = aggregate(scores, n_boot=0)
    return {
        "variant": variant,
        "hardware": f"T4 ({torch.cuda.get_device_name(0)})",
        "size_gb": size_gb,
        "micro_f1": agg["micro_f1"],
        "json_validity": agg["json_validity"],
        "tokens_per_s": round(toks / sum(lat), 2) if sum(lat) else None,
        "mean_latency_s": round(sum(lat) / len(lat), 3) if lat else None,
        "peak_vram_gb": round(peak, 3),
        "n_examples": len(rows),
        "split": split,
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
    }


def awq_quantize(merged_dir: str | Path, out_dir: str | Path, calib_texts: list[str]) -> Path:
    """AutoAWQ 4-bit (GEMM, group size 128). Calibration uses training postings, not test data."""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    model = AutoAWQForCausalLM.from_pretrained(str(merged_dir), safetensors=True)
    tok = AutoTokenizer.from_pretrained(str(merged_dir))
    qcfg = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
    model.quantize(tok, quant_config=qcfg, calib_data=calib_texts, max_calib_seq_len=512)
    out = Path(out_dir)
    model.save_quantized(str(out))
    tok.save_pretrained(str(out))
    return out
