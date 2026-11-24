"""GitHub Actions job matrices for the CPU route, derived from configs/cpu/route.yaml.

python -m jobx.cpu.matrix baselines   # one entry per (model, setting)
python -m jobx.cpu.matrix eval        # one entry per fine-tune quantisation
python -m jobx.cpu.matrix splits      # test splits present in the repo, comma-separated
python -m jobx.cpu.matrix suite [ids] # configs/eval_suite.yaml entries (optionally only the given approach ids)
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


def suite_matrix(only: str = "", root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    """configs/eval_suite.yaml as a job matrix: defaults filled, splits comma-joined, only present splits."""
    import yaml

    cfg = yaml.safe_load((root / "configs/eval_suite.yaml").read_text())
    want = {x.strip() for x in only.split(",") if x.strip()}
    out = []
    for e in cfg["entries"]:
        if want and e["approach"] not in want:
            continue
        splits = [s for s in e["splits"] if (root / SPLIT_FILES[s]).exists()]
        if e["source"] == "hf_convert":
            slug = model_slug(e["model"])
            file = f"{slug}-{e['quant']}.gguf"
            label = e.get("label") or f"{e['model']} (GGUF {e['quant']}, llama.cpp)"
        else:
            slug, file, label = "", e["file"], e.get("label") or e["file"]
        out.append(
            {
                "approach": e["approach"],
                "source": e["source"],
                "model": e.get("model", ""),
                "repo": e.get("repo", ""),
                "slug": slug,
                "quant": e.get("quant", ""),
                "file": file,
                "system": e.get("system", "full"),
                "shots": int(e.get("shots", 0)),
                "guided": bool(e.get("guided", False)),
                "logprobs": bool(e.get("logprobs", False)),
                "splits": ",".join(splits),
                "label": label,
                "ctx": int(cfg.get("ctx", 4096)),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = load_route_config()
    what = argv[0] if argv else "baselines"
    if what == "baselines":
        print(json.dumps({"include": baselines_matrix(cfg)}))
    elif what == "eval":
        print(json.dumps({"include": eval_matrix(cfg)}))
    elif what == "suite":
        print(json.dumps({"include": suite_matrix(argv[1] if len(argv) > 1 else "")}))
    elif what == "splits":
        print(",".join(available_splits()))
    else:
        print(f"unknown matrix {what}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
