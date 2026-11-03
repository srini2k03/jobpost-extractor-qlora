"""Prompts shared by baselines, fine-tuning, serving and the Ollama Modelfile.

Zero-shot, few-shot and fine-tuned runs use the *same* system prompt and user template,
so differences in the results table come from the model/decoding, not the prompt.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from jobx.schema.models import FIELD_ORDER

SYSTEM_PROMPT = """You extract structured data from Indian job postings. Reply with one JSON object and nothing else.

Keys (always all of them, in this order):
company, role, seniority, skills, min_experience_years, max_experience_years, salary, locations, work_mode, employment_type, education

Rules:
- Use null for any field the posting does not state, and [] for empty lists. Never guess.
- company: the hiring company, not a recruitment agency. Drop "Pvt Ltd" style suffixes.
- role: the job title without seniority words ("Sr. Data Engineer" -> "Data Engineer").
- seniority: one of intern, junior, mid, senior, lead, manager, only if stated in words.
- skills: every skill or tool the posting names (job title included), as lower-case canonical names, e.g. "react", "node.js", "power bi", "machine learning".
- experience: numbers in years. "Freshers" -> 0 and 0. "5+ years" -> 5 and null.
- salary: {"min": number, "max": number, "currency": "INR" or "USD", "period": "year", "month" or "hour"} with absolute amounts in that period. "12-18 LPA" -> 1200000 and 1800000 per year; "1.2L/month" -> 120000 per month; "upto 18 LPA" -> min null. Ignore bonuses, kits and fees.
- locations: Indian cities with canonical names (Bengaluru, Gurugram, Mumbai, Thiruvananthapuram ...). Only where the job is based, not the head office.
- work_mode: onsite, hybrid or remote. employment_type: full_time, part_time, contract, internship or freelance.
- education: one of 10th, 12th, diploma, graduate, b.tech, bca, b.sc, b.com, bba, mca, m.tech, m.sc, mba, ca, phd."""

USER_TEMPLATE = "Extract the fields from this job posting.\n\n<posting>\n{text}\n</posting>"


# Short instruction used by the CPU-route LoRA fine-tune. A fine-tuned model learns the conventions from
# its training data, so it does not need the long rule list; the short prompt halves the sequence length
# (cheaper CPU training, faster in-browser inference). Baselines always use SYSTEM_PROMPT.
COMPACT_SYSTEM_PROMPT = (
    "Extract the Indian job posting into one JSON object with keys company, role, seniority, skills, "
    "min_experience_years, max_experience_years, salary, locations, work_mode, employment_type, education. "
    "Use null or [] for anything not stated."
)
SYSTEM_PROMPTS = {"full": SYSTEM_PROMPT, "compact": COMPACT_SYSTEM_PROMPT}


def system_prompt(name: str = "full") -> str:
    return SYSTEM_PROMPTS[name]


def user_prompt(text: str) -> str:
    return USER_TEMPLATE.format(text=text.strip())


def target_json(label: dict[str, Any]) -> str:
    """The exact assistant string the model is trained to produce."""
    ordered = {k: label.get(k) for k in FIELD_ORDER}
    return json.dumps(ordered, ensure_ascii=False, separators=(", ", ": "))


def messages(
    text: str, shots: Sequence[dict[str, Any]] = (), *, system: str = "full"
) -> list[dict[str, str]]:
    """Chat messages for zero-shot (no shots) or few-shot prompting. ``system``: 'full' or 'compact'."""
    msgs = [{"role": "system", "content": system_prompt(system)}]
    for ex in shots:
        msgs.append({"role": "user", "content": user_prompt(ex["text"])})
        msgs.append({"role": "assistant", "content": target_json(ex["label"])})
    msgs.append({"role": "user", "content": user_prompt(text)})
    return msgs


def sft_record(example: dict[str, Any], *, system: str = "full") -> dict[str, Any]:
    """Conversational prompt-completion record for TRL (loss on the completion only)."""
    return {
        "prompt": [
            {"role": "system", "content": system_prompt(system)},
            {"role": "user", "content": user_prompt(example["text"])},
        ],
        "completion": [{"role": "assistant", "content": target_json(example["label"])}],
    }


def select_few_shot(
    train: Sequence[dict[str, Any]], k: int = 3, max_chars: int = 500
) -> list[dict[str, Any]]:
    """Deterministic, diverse few-shot exemplars: short, different families, a salary in each
    of year/month and at least one with null fields."""
    pool = sorted(
        (e for e in train if len(e["text"]) <= max_chars and e["meta"].get("source") == "synthetic"),
        key=lambda e: e["id"],
    )
    chosen: list[dict[str, Any]] = []
    wants = [
        lambda e: e["label"]["salary"] is not None and e["label"]["salary"]["period"] == "year",
        lambda e: e["label"]["salary"] is not None and e["label"]["salary"]["period"] == "month",
        lambda e: e["label"]["company"] is None and e["label"]["salary"] is None,
    ]
    fams: set[str] = set()
    for want in wants[:k]:
        for e in pool:
            if e in chosen or e["meta"]["family"] in fams:
                continue
            if want(e):
                chosen.append(e)
                fams.add(e["meta"]["family"])
                break
    for e in pool:
        if len(chosen) >= k:
            break
        if e not in chosen and e["meta"]["family"] not in fams:
            chosen.append(e)
            fams.add(e["meta"]["family"])
    return chosen[:k]
