"""LLM paraphrase pass for realism (generation runs in notebooks/01_paraphrase_7b.ipynb).

The 7B model only rewrites wording; labels stay the synthetic gold. Every rewrite is checked
by :func:`jobx.data.validate.validate_paraphrase` (rules + optional NLI) and dropped if any
gold value is no longer recoverable or a new fact appeared.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from typing import Any

from jobx.data.validate import NLIScorer, validate_paraphrase

PARAPHRASE_SYSTEM = (
    "You rewrite Indian job postings so they read like a different recruiter wrote them. Keep every fact "
    "exactly: company name, job title, seniority, every skill, experience numbers, salary numbers and units "
    "(LPA, lakhs, per month...), cities, work mode, employment type and education. Do not add any new facts, "
    "numbers, skills, cities, perks or contact details. You may reorder, merge lines, change formatting and "
    "tone (formal, casual, WhatsApp style, Hinglish phrases). Output only the rewritten posting."
)
STYLES: tuple[str, ...] = (
    "formal HR email",
    "casual LinkedIn post",
    "short WhatsApp forward with emojis",
    "a recruiter's Hinglish message",
    "a terse job-board listing",
    "a careers-page description",
)


def paraphrase_messages(text: str, style: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PARAPHRASE_SYSTEM},
        {"role": "user", "content": f"Rewrite this posting as {style}.\n\n<posting>\n{text}\n</posting>"},
    ]


def pick_subset(train: Sequence[dict[str, Any]], n: int, seed: int = 7) -> list[dict[str, Any]]:
    pool = [e for e in train if e["meta"].get("source") == "synthetic"]
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


def strip_wrappers(out: str) -> str:
    s = out.strip()
    for tag in ("<posting>", "</posting>", "```"):
        s = s.replace(tag, "")
    return s.strip()


def filter_paraphrases(
    pairs: Sequence[tuple[dict[str, Any], str, str]],
    *,
    nli: NLIScorer | None = None,
    nli_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """pairs: (source example, style, raw model output). Returns (kept rows, filter stats)."""
    kept: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for src, style, raw in pairs:
        text = strip_wrappers(raw)
        if len(text) < 40:
            reasons["empty_or_too_short"] += 1
            continue
        res = validate_paraphrase(src["text"], text, src["label"], nli=nli, nli_threshold=nli_threshold)
        if not res.ok:
            reasons.update(
                {
                    f.split(":")[0] + (":" + f.split(":")[1] if f.startswith("new:") else ""): 1
                    for f in res.failures
                }
            )
            reasons["_rejected_examples"] += 1
            continue
        kept.append(
            {
                "id": f"para-{src['id']}",
                "text": text,
                "label": src["label"],
                "meta": {
                    **src["meta"],
                    "source": "paraphrase",
                    "family": f"paraphrase/{src['meta']['family']}",
                    "orig_id": src["id"],
                    "style": style,
                    "noise_ops": [],
                },
            }
        )
    n = len(pairs)
    stats = {
        "attempted": n,
        "kept": len(kept),
        "filter_rate": round(1 - len(kept) / n, 4) if n else None,
        "nli": nli is not None,
        "nli_threshold": nli_threshold if nli is not None else None,
        "reasons": dict(reasons.most_common()),
    }
    return kept, stats
