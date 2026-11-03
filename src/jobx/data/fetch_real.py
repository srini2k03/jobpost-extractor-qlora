"""Fetch real public job postings from the Hugging Face Hub for the real-only test split.

Runs where the Hub is reachable (Colab notebook 01 or CI). Keeps postings that look Indian
(an Indian city in the location column or text, or INR/LPA/lakh salary language), 200-3000
characters, de-duplicated, and samples ``n_target`` deterministically.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml

from jobx.normalize.text import loose, matchers

_INDIA_RE = re.compile(r"\b(india|lpa|lakhs?|lacs?|inr|₹|rs\.)", re.IGNORECASE)


def looks_indian(text: str, location: str = "") -> bool:
    if location and (
        re.search(r"\bindia\b", location, re.IGNORECASE) or matchers()["locations"].find(location)
    ):
        return True
    return bool(_INDIA_RE.search(text)) or bool(matchers()["locations"].find(text[:1500]))


def dataset_licence(dataset_id: str) -> str | None:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(dataset_id)
    card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
    lic = None
    if card is not None:
        lic = card.get("license") if isinstance(card, dict) else getattr(card, "license", None)
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    return str(lic).lower() if lic else None


def select_rows(
    rows: Iterable[dict[str, Any]], cand: dict[str, Any], cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    picked: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if i >= cfg.get("max_rows_scanned", 200000):
            break
        text = next((str(row[c]) for c in cand["text_columns"] if row.get(c)), "")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not (cfg["min_chars"] <= len(text) <= cfg["max_chars"]):
            continue
        loc = " ".join(str(row.get(c) or "") for c in cand.get("location_columns", []))
        if not looks_indian(text, loc):
            continue
        h = hashlib.sha1(loose(text).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        picked.append(
            {
                "row": i,
                "text": text,
                "location_field": loc or None,
                "company_field": row.get(cand.get("company_column") or "", None),
                "title_field": row.get(cand.get("title_column") or "", None),
            }
        )
    rng = random.Random(cfg.get("seed", 11))
    rng.shuffle(picked)
    return sorted(picked[: cfg["n_target"]], key=lambda r: r["row"])


def fetch(
    config_path: str | Path, out_path: str | Path, *, allow_unverified_licence: bool = False
) -> dict[str, Any]:
    from datasets import load_dataset

    cfg = yaml.safe_load(Path(config_path).read_text())
    allow = set(cfg["licence_allowlist"])
    log: list[dict[str, Any]] = []
    for cand in cfg["candidates"]:
        try:
            lic = dataset_licence(cand["id"])
        except Exception as e:  # dataset missing or private
            log.append({"id": cand["id"], "skipped": f"not reachable: {type(e).__name__}"})
            continue
        if lic not in allow and not allow_unverified_licence:
            log.append({"id": cand["id"], "skipped": f"licence {lic!r} not in allowlist"})
            continue
        ds = load_dataset(cand["id"], split=cand.get("split", "train"), streaming=True)
        rows = select_rows(_iter(ds), cand, cfg)
        if len(rows) < 50:
            log.append({"id": cand["id"], "skipped": f"only {len(rows)} Indian postings found"})
            continue
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for k, r in enumerate(rows):
                f.write(
                    json.dumps(
                        {"id": f"real-{k:04d}", **r, "dataset": cand["id"], "licence": lic},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        meta = {
            "dataset": cand["id"],
            "licence_declared": lic,
            "licence_verified_by_human": False,
            "n": len(rows),
            "candidates_log": log,
        }
        out.with_name("raw_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        return meta
    raise RuntimeError(f"no candidate dataset usable: {log}")


def _iter(ds: Any) -> Iterator[dict[str, Any]]:
    yield from ds
