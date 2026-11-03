"""GitHub Actions job matrices for the CPU route, derived from configs/cpu/route.yaml.

python -m jobx.cpu.matrix baselines   # one entry per (model, setting)
python -m jobx.cpu.matrix eval        # one entry per fine-tune quantisation
python -m jobx.cpu.matrix splits      # test splits present in the repo, comma-separated
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from jobx import REPO_ROOT
from jobx.cpu.train import load_route_config
from jobx.eval.run import SPLIT_FILES


def available_splits(root: Path = REPO_ROOT) -> list[str]:
    return [s for s in ("test_synth", "test_real") if (root / SPLIT_FILES[s]).exists()]


def model_slug(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1].lower()


def baselines_matrix(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    b = cfg["baselines"]
    out = []
    for m in b["models"]:
        for s in m["settings"]:
            guided = s.endswith("_guided")
            out.append(
                {
                    "model": m["model"],
                    "slug": model_slug(m["model"]),
                    "approach": f"{m['prefix']}_{s}",
                    "shots": 3 if s.startswith("few_shot") else 0,
                    "guided": guided,
                    "max_examples": (b["guided_max"] if guided else b.get("free_max")) or 0,
                    "quant": b["quant"],
                }
            )
    return out


def eval_matrix(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    e = cfg["eval"]
    return [
        {"quant": q, "approach": f"{e['prefix']}_{q.lower()}", "variant": f"LoRA 0.5B GGUF {q} (llama.cpp)"}
        for q in e["quants"]
    ]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = load_route_config()
    what = argv[0] if argv else "baselines"
    if what == "baselines":
        print(json.dumps({"include": baselines_matrix(cfg)}))
    elif what == "eval":
        print(json.dumps({"include": eval_matrix(cfg)}))
    elif what == "splits":
        print(",".join(available_splits()))
    else:
        print(f"unknown matrix {what}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
