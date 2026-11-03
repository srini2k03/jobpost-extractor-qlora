"""Noise operators that make rendered postings look scraped/forwarded.

Every operator preserves the meaning of the gold fields: words that belong to a gold surface
string ("protected" words) are never typo'd, and inserted distractors are worded so they are
clearly not the salary/experience/location of the job. ``tests/test_generate.py`` checks this
by running the recoverability validator on the generated set.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable

from jobx import vocab
from jobx.data.templates import STREETS

ABBREV: dict[str, str] = {
    "experience": "exp",
    "years": "yrs",
    "required": "reqd",
    "with": "w/",
    "and": "&",
    "candidates": "candts",
    "immediate": "immdt",
    "please": "pls",
    "opportunity": "oppty",
    "requirement": "reqmt",
    "knowledge": "knowldg",
    "months": "mnths",
    "minimum": "min",
}
SAFE_IN_PROTECTED = frozenset({"years", "experience", "minimum"})
_CASE_SENSITIVE = (
    vocab.CASE_SENSITIVE_ALIASES
    | vocab.CASE_SENSITIVE_CITY_ALIASES
    | vocab.CASE_SENSITIVE_WORK_MODE
    | vocab.CASE_SENSITIVE_EDUCATION
)


def _phone(rng: random.Random) -> str:
    return f"+91 {rng.randint(6, 9)}{rng.randint(100000000, 999999999)}"


DISTRACTORS: tuple[Callable[[random.Random], str], ...] = (
    lambda r: f"📞 Call/WhatsApp HR: {_phone(r)}",
    lambda r: "Joining kit worth ₹500 for all new joiners",
    lambda r: f"Employee referral bonus of ₹{r.choice([2000, 5000, 10000])} after joining",
    lambda r: f"Notice period: {r.choice([15, 30, 45, 60, 90])} days",
    lambda r: f"Openings: {r.randint(2, 60)}",
    lambda r: f"Founded in {r.randint(1995, 2021)} | {r.choice([50, 200, 500, 1000, 5000])}+ employees",
    lambda r: f"Apply before {r.randint(1, 28)}/{r.randint(1, 12)}/2025",
    lambda r: "Shift timing: 9:30 AM - 6:30 PM",
    lambda r: f"Job code: JX-{r.randint(1000, 9999)}",
    lambda r: f"Interview rounds: {r.randint(2, 4)}",
    lambda r: f"We have been in business for over {r.choice([10, 12, 15, 20])} years.",
    lambda r: f"Office: {r.randint(2, 15)}th floor, {r.choice(STREETS)}, PIN {r.randint(110001, 799999)}",
    lambda r: f"Contact: hr{r.randint(1, 99)}@careers.example.in",
)


def _lines(text: str) -> list[str]:
    return text.split("\n")


def op_abbreviate(text: str, rng: random.Random, protected: set[str]) -> str:
    def sub(m: re.Match[str]) -> str:
        w = m.group(0)
        lw = w.lower()
        if lw in ABBREV and (lw not in protected or lw in SAFE_IN_PROTECTED) and rng.random() < 0.7:
            rep = ABBREV[lw]
            return rep.upper() if w.isupper() else (rep.capitalize() if w[0].isupper() else rep)
        return w

    return re.sub(r"[A-Za-z]+", sub, text)


def _typo(w: str, rng: random.Random) -> str:
    i = rng.randrange(1, len(w) - 1)
    kind = rng.randrange(3)
    if kind == 0:
        return w[:i] + w[i + 1] + w[i] + w[i + 2 :]
    if kind == 1:
        return w[:i] + w[i + 1 :]
    return w[:i] + w[i] + w[i:]


def op_typos(text: str, rng: random.Random, protected: set[str]) -> str:
    rate = rng.uniform(0.03, 0.08)

    def sub(m: re.Match[str]) -> str:
        w = m.group(0)
        if len(w) >= 5 and w.lower() not in protected and w.lower() not in ABBREV and rng.random() < rate:
            return _typo(w, rng)
        return w

    return re.sub(r"(?<![A-Za-z0-9.@/#])[A-Za-z]+(?![A-Za-z0-9.@/])", sub, text)


def op_emojis(text: str, rng: random.Random, protected: set[str]) -> str:
    out = []
    for ln in _lines(text):
        if ln.strip() and rng.random() < 0.35:
            ln = f"{rng.choice(vocab.EMOJIS)} {ln}"
        out.append(ln)
    return "\n".join(out)


def op_bullets(text: str, rng: random.Random, protected: set[str]) -> str:
    b = rng.choice(vocab.BULLETS)
    return "\n".join(
        f"{b} {ln}" if ln.strip() and i > 0 and rng.random() < 0.6 else ln
        for i, ln in enumerate(_lines(text))
    )


def op_caps_header(text: str, rng: random.Random, protected: set[str]) -> str:
    ls = _lines(text)
    if ls and not any(cs in ls[0] for cs in _CASE_SENSITIVE):
        ls[0] = ls[0].upper()
    return "\n".join(ls)


def op_whitespace(text: str, rng: random.Random, protected: set[str]) -> str:
    out = []
    for ln in _lines(text):
        if rng.random() < 0.2:
            ln = ln.replace(": ", " :  ", 1)
        if rng.random() < 0.2:
            ln += "   "
        out.append(ln)
        if rng.random() < 0.1:
            out.append("")
    return "\n".join(out)


def op_html(text: str, rng: random.Random, protected: set[str]) -> str:
    text = text.replace("&", "&amp;")
    return text.replace("\n", rng.choice(["<br>", "<br/>\n", "<br />"]))


def op_single_line(text: str, rng: random.Random, protected: set[str]) -> str:
    sep = rng.choice([" | ", " ", " ; ", " • "])
    return sep.join(ln.strip() for ln in _lines(text) if ln.strip())


def op_hinglish(text: str, rng: random.Random, protected: set[str]) -> str:
    ls = _lines(text)
    ls.insert(rng.randint(1, max(1, len(ls))), rng.choice(vocab.HINGLISH_FRAGMENTS))
    return "\n".join(ls)


def op_distractors(text: str, rng: random.Random, protected: set[str]) -> str:
    ls = _lines(text)
    for d in rng.sample(DISTRACTORS, rng.randint(1, 3)):
        ls.insert(rng.randint(1, max(1, len(ls))), d(rng))
    return "\n".join(ls)


def op_shuffle_sections(text: str, rng: random.Random, protected: set[str]) -> str:
    parts = text.split("\n\n")
    if len(parts) < 3:
        return text
    head, rest = parts[0], parts[1:]
    rng.shuffle(rest)
    return "\n\n".join([head, *rest])


Op = Callable[[str, random.Random, set[str]], str]
# (name, op, probability). Order matters: structure-changing ops run last.
OPS: tuple[tuple[str, Op, float], ...] = (
    ("shuffle_sections", op_shuffle_sections, 0.3),
    ("distractors", op_distractors, 0.55),
    ("hinglish", op_hinglish, 0.15),
    ("abbreviate", op_abbreviate, 0.35),
    ("typos", op_typos, 0.35),
    ("emojis", op_emojis, 0.2),
    ("bullets", op_bullets, 0.15),
    ("caps_header", op_caps_header, 0.12),
    ("whitespace", op_whitespace, 0.2),
    ("single_line", op_single_line, 0.1),
    ("html", op_html, 0.06),
)


def apply_noise(
    text: str, rng: random.Random, protected: set[str], *, sectioned: bool
) -> tuple[str, list[str]]:
    applied: list[str] = []
    for name, op, p in OPS:
        if name == "shuffle_sections" and not sectioned:
            continue
        if name == "html" and "single_line" in applied:
            continue
        if rng.random() < p:
            text = op(text, rng, protected)
            applied.append(name)
    return text.strip(), applied
