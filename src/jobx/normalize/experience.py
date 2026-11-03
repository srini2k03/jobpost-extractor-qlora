"""Experience-range parsing ("3-5 yrs", "5+ years", "Freshers", "min 2 years")."""

from __future__ import annotations

import re
from dataclasses import dataclass

_YR = r"(?:years?|yrs?\.?|yr\.?|y(?![a-z]))"
_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twelve": 12,
    "fifteen": 15,
}
_N = r"(\d{1,2}(?:\.\d)?|" + "|".join(_WORDS) + r")"
_RANGE = re.compile(rf"(?<![A-Za-z\d.]){_N}\s*(?:-|–|to)\s*{_N}\s*\+?\s*{_YR}", re.IGNORECASE)
_PLUS = re.compile(rf"(?<![A-Za-z\d.]){_N}\s*\+\s*{_YR}", re.IGNORECASE)
_MIN = re.compile(rf"(?:min(?:imum)?\.?|at\s*least)\s*(?:of\s*)?{_N}\s*{_YR}", re.IGNORECASE)
_LABEL = re.compile(
    rf"(?:exp(?:erience)?\.?|work\s*ex)\s*(?:required|reqd\.?)?\s*[:\-–]?\s*{_N}\s*(?:-|–|to)\s*{_N}(?!\s*(?:lpa|l|k|lakh))",
    re.IGNORECASE,
)
_UPTO = re.compile(rf"(?:up\s*to|upto|max(?:imum)?)\s*{_N}\s*{_YR}", re.IGNORECASE)
_FRESHER = re.compile(r"\bfreshers?\b(?!\s*(?:and|&|/|or)\s*experienced)", re.IGNORECASE)


def _val(s: str) -> float:
    s = s.lower()
    return float(_WORDS[s]) if s in _WORDS else float(s)


@dataclass(frozen=True)
class ExpMention:
    min: float | None
    max: float | None
    start: int
    end: int


def find_experience(text: str) -> list[ExpMention]:
    out: list[ExpMention] = []
    for rx, kind in ((_RANGE, "range"), (_LABEL, "range"), (_PLUS, "min"), (_MIN, "min"), (_UPTO, "max")):
        for m in rx.finditer(text):
            if any(m.start() < o.end and o.start < m.end() for o in out):
                continue
            a = _val(m.group(1))
            if kind == "range":
                b = _val(m.group(2))
                if b < a or b > 40:
                    continue
                out.append(ExpMention(a, b, m.start(), m.end()))
            elif kind == "min":
                out.append(ExpMention(a, None, m.start(), m.end()))
            else:
                out.append(ExpMention(None, a, m.start(), m.end()))
    for m in _FRESHER.finditer(text):
        out.append(ExpMention(0.0, 0.0, m.start(), m.end()))
    out.sort(key=lambda x: x.start)
    return out
