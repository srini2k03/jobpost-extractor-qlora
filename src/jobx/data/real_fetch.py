"""Fetch REAL job postings for the held-out real-world test set from licence-checked Hub datasets.

Sources and their filters live in ``configs/real_data.yaml`` (``sources:``). Each source is pinned to a
dataset revision; its licence is re-read from the card at fetch time and must be in the allowlist.
Rows are filtered to Indian postings by the dataset's own country / location field, composed into one
posting text (format documented per source in ``compose``), length-filtered, exact-deduplicated and
sampled with a fixed seed. The output pool keeps the source row id, revision and licence of every
posting, so each one can be traced and attributed.

    python -m jobx.data.real_fetch --out data/real/pool_hf.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from jobx.data.real_probe import card_licence, licence_from_tags
from jobx.normalize.text import loose


def _s(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def compose(kind: str, row: dict[str, Any]) -> str:
    """One posting text from a dataset row. Only real field values are used; the few line labels added
    ("Location:", "Employment type:", "Salary:") mirror what the source's posting page displays."""
    if kind == "emscad":
        parts = [_s(row.get("title"))]
        if _s(row.get("location")):
            parts.append(f"Location: {_s(row.get('location'))}")
        if _s(row.get("employment_type")):
            parts.append(f"Employment type: {_s(row.get('employment_type'))}")
        if _s(row.get("salary_range")):
            parts.append(f"Salary: {_s(row.get('salary_range'))}")
        body = [_s(row.get(k)) for k in ("description", "requirements", "benefits") if _s(row.get(k))]
        return _clean("\n".join(parts) + "\n\n" + "\n\n".join(body))
    if kind == "joa":
        head = [
            _s(row.get("title")),
            " · ".join(x for x in (_s(row.get("employer")), _s(row.get("location"))) if x),
        ]
        return _clean("\n".join(h for h in head if h) + "\n\n" + _s(row.get("description")))
    if kind == "ats":
        loc = _s(row.get("location_raw"))
        head = [_s(row.get("title")), " · ".join(x for x in (_s(row.get("company")), loc) if x)]
        return _clean("\n".join(h for h in head if h) + "\n\n" + _s(row.get("description_text")))
    raise ValueError(f"unknown source kind {kind!r}")


def is_indian(kind: str, row: dict[str, Any]) -> bool:
    """India by the dataset's own country field (never by the bare 'IN' of US state codes)."""
    if kind == "emscad":
        return _s(row.get("location")).upper().startswith("IN,") and _s(row.get("fraudulent")) in (
            "0",
            "False",
        )
    if kind in ("joa", "ats"):
        return _s(row.get("country")).upper() in ("IN", "IND", "INDIA")
    raise ValueError(kind)


def row_id(kind: str, row: dict[str, Any], index: int) -> str:
    for k in ("job_id", "id"):
        if _s(row.get(k)):
            return _s(row.get(k))
    return str(index)


def select(
    rows: Iterable[dict[str, Any]],
    src: dict[str, Any],
    *,
    min_chars: int,
    max_chars: int,
    seed: int,
    seen: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kind = src["kind"]
    seen = seen if seen is not None else set()
    picked: list[dict[str, Any]] = []
    c: Counter[str] = Counter()
    for i, row in enumerate(rows):
        c["rows"] += 1
        if not is_indian(kind, row):
            continue
        c["india"] += 1
        text = compose(kind, row)
        if not (min_chars <= len(text) <= max_chars):
            c["length"] += 1
            continue
        h = hashlib.sha1(loose(text).encode()).hexdigest()
        if h in seen:
            c["duplicate"] += 1
            continue
        seen.add(h)
        picked.append({"text": text, "source_row": row_id(kind, row, i)})
    rng = random.Random(seed)
    rng.shuffle(picked)
    c["eligible"] = len(picked)
    return picked[: int(src["n"])], dict(c)


def fetch_sources(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    allow = set(cfg["licence_allowlist"])
    api = HfApi()
    out: list[dict[str, Any]] = []
    log: dict[str, Any] = {}
    seen: set[str] = set()
    for src in cfg["sources"]:
        ds_id, rev = src["id"], src.get("revision")
        info = api.dataset_info(ds_id, revision=rev)
        lic = card_licence(getattr(info, "card_data", None)) or licence_from_tags(getattr(info, "tags", None))
        if lic not in allow:
            log[ds_id] = {"skipped": f"licence {lic!r} not in allowlist"}
            continue
        ds = load_dataset(ds_id, split=src.get("split", "train"), revision=info.sha, streaming=True)
        picked, counts = select(
            ds,
            src,
            min_chars=int(cfg["min_chars"]),
            max_chars=int(cfg["max_chars"]),
            seed=int(cfg["seed"]),
            seen=seen,
        )
        for k, r in enumerate(picked):
            out.append(
                {
                    "id": f"real-{src['prefix']}-{k:03d}",
                    "text": r["text"],
                    "source": {
                        "dataset": ds_id,
                        "revision": info.sha,
                        "licence": lic,
                        "row_id": r["source_row"],
                        "composition": src["kind"],
                    },
                }
            )
        log[ds_id] = {"licence": lic, "revision": info.sha, **counts, "picked": len(picked)}
        print(ds_id, log[ds_id])
    return out, log


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/real_data.yaml")
    ap.add_argument("--out", default="data/real/pool_hf.jsonl")
    ap.add_argument("--log", default="results/real/fetch_hf.json")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    rows, log = fetch_sources(cfg)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    Path(a.log).write_text(json.dumps({"sources": log, "n": len(rows)}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
