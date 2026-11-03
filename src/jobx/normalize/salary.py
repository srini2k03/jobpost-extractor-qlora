"""Salary parsing and normalisation to INR per annum.

Handles the formats Indian postings actually use: "₹12–18 LPA", "CTC 8-10 lakhs",
"1.2L/month", "15k-25k p.m.", "INR 12,00,000 - 18,00,000 p.a.", "upto 18 LPA",
"$40k-60k/yr". Used by the evaluator (normalise predictions and gold), the regex baseline
and the paraphrase validator.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from jobx.vocab import HOURS_PER_YEAR, USD_TO_INR

PERIOD_FACTOR = {"year": 1.0, "month": 12.0, "hour": HOURS_PER_YEAR}

_NUM = r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_UNIT = r"(?:lpa|lakhs?|lacs?|lakh|crores?|cr|k|l)(?![a-z])"
_CUR = r"(?:₹|rs\.?|inr|\$|usd)"
_POST = (
    r"(?:per\s+annum|per\s+year|p\.\s?a\.?|pa(?![a-z])|/\s*(?:yr|year|annum)\b|annually|"
    r"per\s+month|p\.\s?m\.?|pm(?![a-z])|/\s*(?:month|mo|m)\b|monthly|a\s+month)"
)
SALARY_RE = re.compile(
    rf"(?<![A-Za-z0-9.,])(?P<cur1>{_CUR})?\s*(?P<n1>{_NUM})\s*(?P<u1>{_UNIT})?"
    rf"(?:\s*(?:-|–|—|to)\s*(?P<cur2>{_CUR})?\s*(?P<n2>{_NUM})\s*(?P<u2>{_UNIT})?)?"
    rf"(?P<post>\s*(?:/-)?\s*(?:{_POST}))?",
    re.IGNORECASE,
)
_KEYWORD_RE = re.compile(
    r"(salary|ctc|package|stipend|pay\b|compensation|budget|remuneration|in-hand|in hand|lpa|tankha)",
    re.IGNORECASE,
)
_UPTO_RE = re.compile(r"(up\s*to|upto|max(?:imum)?\.?|till|not more than)\s*[:\-]?\s*$", re.IGNORECASE)
_MIN_RE = re.compile(
    r"(min(?:imum)?\.?|starting(?:\s+(?:from|at))?|from|at\s+least)\s*[:\-]?\s*$", re.IGNORECASE
)

ANNUAL_RANGE = (50_000.0, 2e8)
MONTHLY_RANGE = (3_000.0, 2e6)


@dataclass(frozen=True)
class SalaryMention:
    min: float | None
    max: float | None
    currency: str
    period: str
    start: int
    end: int

    def per_annum_inr(self) -> tuple[float | None, float | None]:
        return to_inr_per_annum(
            {"min": self.min, "max": self.max, "currency": self.currency, "period": self.period}
        ) or (None, None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "min": _clean(self.min),
            "max": _clean(self.max),
            "currency": self.currency,
            "period": self.period,
        }


def _clean(v: float | None) -> float | int | None:
    if v is None:
        return None
    return round(v) if abs(v - round(v)) < 1e-6 else round(v, 2)


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _unit_mult(u: str | None) -> float:
    if not u:
        return 1.0
    u = u.lower()
    if u == "k":
        return 1e3
    if u.startswith("cr"):
        return 1e7
    return 1e5  # lpa / lakh / lac / l


def _period_from(u1: str | None, u2: str | None, post: str | None) -> str | None:
    units = " ".join(x.lower() for x in (u1, u2) if x)
    p = (post or "").lower()
    if any(t in p for t in ("month", "p.m", "p. m", "mo", "/m")) or re.search(r"\bpm\b", p):
        return "month"
    if p.strip() or "lpa" in units:
        return "year"
    return None


def find_salaries(text: str) -> list[SalaryMention]:
    """All plausible salary mentions in ``text`` (distractors such as phone numbers,
    dates or "₹500 joining kit" are rejected by context and plausibility checks)."""
    out: list[SalaryMention] = []
    for m in SALARY_RE.finditer(text):
        n1, n2 = m.group("n1"), m.group("n2")
        u1, u2, post = m.group("u1"), m.group("u2"), m.group("post")
        cur = (m.group("cur1") or m.group("cur2") or "").lower()
        before = text[max(0, m.start() - 40) : m.start()]
        has_unit = bool(u1 or u2 or post)
        if not has_unit and not _KEYWORD_RE.search(before):
            continue
        if u1 is None and u2 is not None and n2 is not None:
            u1 = u2  # "12-18 LPA": unit applies to both ends
        if u2 is None and u1 is not None and n2 is not None:
            u2 = u1
        a = _num(n1) * _unit_mult(u1)
        b = _num(n2) * _unit_mult(u2) if n2 is not None else None
        period = _period_from(u1, u2, post)
        if period is None:
            lakhish = any(x and x.lower() not in ("k",) for x in (u1, u2))
            period = "year" if lakhish or max(a, b or 0) >= 100_000 else "month"
        currency = "USD" if cur in ("$", "usd") else "INR"
        lo_ok, hi_ok = ANNUAL_RANGE if period == "year" else MONTHLY_RANGE
        if currency == "USD":
            lo_ok, hi_ok = lo_ok / USD_TO_INR / 4, hi_ok / USD_TO_INR
        vals = [v for v in (a, b) if v is not None]
        if not all(lo_ok <= v <= hi_ok for v in vals):
            continue
        lo: float | None = a
        hi: float | None = b if b is not None else a
        if b is None:
            prefix = text[max(0, m.start() - 20) : m.start()]
            if _UPTO_RE.search(prefix):
                lo, hi = None, a
            elif _MIN_RE.search(prefix) or text[m.end() : m.end() + 1] == "+":
                lo, hi = a, None
        if lo is not None and hi is not None and lo > hi:
            lo, hi = hi, lo
        out.append(SalaryMention(lo, hi, currency, period, m.start(), m.end()))
    return out


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, str):
        ms = find_salaries(v if _KEYWORD_RE.search(v) else f"salary {v}")
        if ms:
            m = ms[0]
            val = m.min if m.min is not None else m.max
            return val
        try:
            return float(v.replace(",", "").strip())
        except ValueError:
            return None
    return None


def to_inr_per_annum(salary: Any) -> tuple[float | None, float | None] | None:
    """Normalise a salary (schema dict or free string) to (min, max) INR per annum.

    Returns None when there is no usable salary. Strings are parsed with :func:`find_salaries`.
    """
    if salary is None:
        return None
    if isinstance(salary, str):
        ms = find_salaries(salary if _KEYWORD_RE.search(salary) else f"salary {salary}")
        if not ms:
            return None
        return to_inr_per_annum(ms[0].as_dict())
    if not isinstance(salary, Mapping):
        return None
    lo, hi = _to_float(salary.get("min")), _to_float(salary.get("max"))
    if lo is None and hi is None:
        return None
    period = str(salary.get("period") or "year").lower()
    period = {"annum": "year", "annual": "year", "yearly": "year", "monthly": "month", "hourly": "hour"}.get(
        period, period
    )
    factor = PERIOD_FACTOR.get(period, 1.0)
    currency = str(salary.get("currency") or "INR").upper()
    fx = USD_TO_INR if currency == "USD" else 1.0
    return (
        None if lo is None else lo * factor * fx,
        None if hi is None else hi * factor * fx,
    )


def salary_key(salary: Any) -> tuple[int | None, int | None] | None:
    """Comparable key: INR per annum rounded to the nearest thousand."""
    pa = to_inr_per_annum(salary)
    if pa is None:
        return None
    lo, hi = pa
    return (None if lo is None else round(lo / 1000), None if hi is None else round(hi / 1000))
