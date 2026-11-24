"""Field-level evidence: where in the posting each extracted value is supported.

For every non-null value (every element of a list field) :func:`find_evidence` returns the supporting
character span(s) in the *input* text and how they were found:

======== ================================================================================================
match    meaning
======== ================================================================================================
exact    the value occurs verbatim (case-insensitive, on word boundaries)
alias    an alias of the canonical value occurs ("Bangalore" for "Bengaluru", "WFH" for "remote")
parsed   the salary / experience parser reads the same normalised value from a mention in the text
normal.  the value's word tokens occur contiguously after normalisation (punctuation, legal suffixes,
         seniority words dropped), e.g. "Kaveri Analytics" in "Kaveri Analytics Pvt. Ltd."
partial  a parsed mention agrees on part of the value (one salary/experience bound, or a number only)
fuzzy    a window of the text is within edit-distance ratio >= ``FUZZY_MIN`` of the value (typos)
none     no support found: the value is flagged ``unsupported``
======== ================================================================================================

``status`` is ``supported`` (exact / alias / parsed / normalised), ``weak`` (partial / fuzzy) or
``unsupported``. Spans are never synthesised: every span is a slice of the input and its ``text`` is
exactly ``text[start:end]``. :func:`missing_cues` does the opposite check for null fields: it reports
text that looks like the field was stated (a salary or experience mention, a known city...), which is a
signal for a possible omission rather than proof of one.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass
from dataclasses import field as dc_field
from typing import Any, Literal

from jobx.normalize.experience import find_experience
from jobx.normalize.fields import norm_company, norm_number, norm_role
from jobx.normalize.salary import find_salaries, salary_key, to_inr_per_annum
from jobx.normalize.text import matchers
from jobx.schema.models import FIELD_ORDER, LIST_FIELDS

Status = Literal["supported", "weak", "unsupported"]
Match = Literal["exact", "alias", "parsed", "normalised", "partial", "fuzzy", "none"]

FUZZY_MIN = 0.85
SUPPORTED: frozenset[str] = frozenset({"exact", "alias", "parsed", "normalised"})
WEAK: frozenset[str] = frozenset({"partial", "fuzzy"})
ALIAS_FIELDS = {"skills": "skills", "locations": "locations", "work_mode": "work_mode"}
ALIAS_FIELDS |= {"employment_type": "employment_type", "education": "education"}

_TOKEN = re.compile(r"[A-Za-z0-9+#]+")
_SENIORITY_CUES: dict[str, str] = {
    "intern": r"\bintern(ship)?\b",
    "junior": r"\b(junior|jr\.?|trainee|entry[- ]level|fresher)\b",
    "mid": r"\b(mid[- ]?(level|senior)?|intermediate|II)\b",
    "senior": r"\b(senior|sr\.?)\b",
    "lead": r"\b(lead|principal|staff)\b",
    "manager": r"\b(manager|managerial|head of)\b",
}


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str


@dataclass
class Evidence:
    field: str
    value: Any
    status: Status
    match: Match
    spans: list[Span] = dc_field(default_factory=list)
    score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _span(text: str, s: int, e: int) -> Span:
    return Span(s, e, text[s:e])


def _status(match: Match) -> Status:
    if match in SUPPORTED:
        return "supported"
    if match in WEAK:
        return "weak"
    return "unsupported"


def _tokens(text: str) -> list[tuple[str, int, int]]:
    return [(m.group().lower(), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def find_exact(text: str, value: str) -> list[Span]:
    v = value.strip()
    if len(v) < 2:
        return []
    pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(v)}(?![A-Za-z0-9])", re.IGNORECASE)
    return [_span(text, m.start(), m.end()) for m in pat.finditer(text)]


def find_token_sequence(text: str, value: str, toks: list[tuple[str, int, int]] | None = None) -> list[Span]:
    """Contiguous occurrences of the value's word tokens (normalised: case and punctuation ignored)."""
    want = [t for t, _, _ in _tokens(value)]
    if not want:
        return []
    toks = toks if toks is not None else _tokens(text)
    words = [t for t, _, _ in toks]
    k = len(want)
    out = []
    for i in range(len(words) - k + 1):
        if words[i : i + k] == want:
            out.append(_span(text, toks[i][1], toks[i + k - 1][2]))
    return out


def find_fuzzy(
    text: str, value: str, toks: list[tuple[str, int, int]] | None = None, *, min_ratio: float = FUZZY_MIN
) -> tuple[Span | None, float]:
    """Best window of k-1..k+1 tokens whose lower-cased text is closest to the value."""
    target = " ".join(t for t, _, _ in _tokens(value))
    if len(target) < 4:
        return None, 0.0
    toks = toks if toks is not None else _tokens(text)
    k = len(target.split())
    best: tuple[float, int, int] = (0.0, -1, -1)
    sm = difflib.SequenceMatcher(autojunk=False)
    sm.set_seq2(target)
    for w in {max(1, k - 1), k, k + 1}:
        for i in range(len(toks) - w + 1):
            cand = " ".join(t for t, _, _ in toks[i : i + w])
            if abs(len(cand) - len(target)) > max(3, len(target) // 3):
                continue
            sm.set_seq1(cand)
            if sm.real_quick_ratio() < min_ratio or sm.quick_ratio() < min_ratio:
                continue
            r = sm.ratio()
            if r > best[0]:
                best = (r, toks[i][1], toks[i + w - 1][2])
    if best[0] >= min_ratio:
        return _span(text, best[1], best[2]), round(best[0], 3)
    return None, round(best[0], 3)


def _string_evidence(
    text: str, fname: str, value: str, toks: list[tuple[str, int, int]], alias_key: str | None
) -> Evidence:
    if alias_key:
        hits = [h for h in matchers()[alias_key].find(text) if h.canonical == value]
        if hits:
            spans = [_span(text, h.start, h.end) for h in hits]
            exact = any(s.text.lower() == value.lower() for s in spans)
            return Evidence(fname, value, "supported", "exact" if exact else "alias", spans, 1.0)
    spans = find_exact(text, value)
    if spans:
        return Evidence(fname, value, "supported", "exact", spans, 1.0)
    variants = [value]
    if fname == "company":
        variants.append(norm_company(value) or "")
    if fname == "role":
        variants.append(norm_role(value) or "")
    for v in variants:
        spans = find_token_sequence(text, v, toks) if v else []
        if spans:
            return Evidence(fname, value, "supported", "normalised", spans, 1.0)
    sp, ratio = find_fuzzy(text, value, toks)
    if sp is not None:
        return Evidence(fname, value, "weak", "fuzzy", [sp], ratio)
    return Evidence(fname, value, "unsupported", "none", [], ratio)


def _number_spans(text: str, numbers: set[float]) -> list[Span]:
    out = []
    for m in re.finditer(r"(?<![\d.,])\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|(?<![\d.,])\d+(?:\.\d+)?", text):
        try:
            v = float(m.group().replace(",", ""))
        except ValueError:
            continue
        if any(abs(v - n) < 1e-6 for n in numbers):
            out.append(_span(text, m.start(), m.end()))
    return out


def _salary_evidence(text: str, value: dict[str, Any]) -> Evidence:
    key = salary_key(value)
    mentions = find_salaries(text)
    same = [m for m in mentions if salary_key(m.as_dict()) == key]
    if same:
        return Evidence(
            "salary", value, "supported", "parsed", [_span(text, m.start, m.end) for m in same], 1.0
        )
    pa = to_inr_per_annum(value) or (None, None)
    bounds = {round(v / 1000) for v in pa if v is not None}
    partial = [m for m in mentions if bounds & {round(v / 1000) for v in m.per_annum_inr() if v is not None}]
    if partial:
        return Evidence(
            "salary", value, "weak", "partial", [_span(text, m.start, m.end) for m in partial], 0.5
        )
    nums: set[float] = set()
    for k in ("min", "max"):
        v = norm_number(value.get(k))
        if v:
            nums |= {v, v / 1e3, v / 1e5, v / 1e7}
    spans = _number_spans(text, nums)
    if spans:
        return Evidence("salary", value, "weak", "partial", spans[:3], 0.3)
    return Evidence("salary", value, "unsupported", "none", [], 0.0)


def _experience_evidence(text: str, fname: str, lo: float | None, hi: float | None) -> Evidence:
    v = lo if fname == "min_experience_years" else hi
    mentions = find_experience(text)
    full = [m for m in mentions if (m.min, m.max) == (lo, hi)]
    if full:
        return Evidence(fname, v, "supported", "parsed", [_span(text, m.start, m.end) for m in full], 1.0)
    bound = [m for m in mentions if (m.min if fname == "min_experience_years" else m.max) == v]
    if bound:
        return Evidence(fname, v, "weak", "partial", [_span(text, m.start, m.end) for m in bound], 0.5)
    if v is not None:
        spans = [
            s
            for s in _number_spans(text, {float(v)})
            if re.match(r"\s*(?:\+|-|–|to)?\s*\d*\s*\+?\s*(?:years?|yrs?)", text[s.end : s.end + 16], re.I)
        ]
        if spans:
            return Evidence(fname, v, "weak", "partial", spans[:2], 0.3)
    return Evidence(fname, v, "unsupported", "none", [], 0.0)


def _seniority_evidence(text: str, value: str) -> Evidence:
    cue = _SENIORITY_CUES.get(value)
    if cue:
        spans = [_span(text, m.start(), m.end()) for m in re.finditer(cue, text, re.IGNORECASE)]
        if spans:
            return Evidence("seniority", value, "supported", "alias", spans[:3], 1.0)
    return Evidence("seniority", value, "unsupported", "none", [], 0.0)


def find_evidence(text: str, record: dict[str, Any] | None) -> list[Evidence]:
    """Evidence for every non-null value in ``record`` (one entry per list element)."""
    if not record:
        return []
    toks = _tokens(text)
    out: list[Evidence] = []
    lo = norm_number(record.get("min_experience_years"))
    hi = norm_number(record.get("max_experience_years"))
    for f in FIELD_ORDER:
        v = record.get(f)
        if v is None or v == [] or v == "":
            continue
        if f in LIST_FIELDS:
            for el in v if isinstance(v, list) else [v]:
                if isinstance(el, str) and el.strip():
                    out.append(_string_evidence(text, f, el, toks, ALIAS_FIELDS[f]))
        elif f == "salary" and isinstance(v, dict):
            out.append(_salary_evidence(text, v))
        elif f in ("min_experience_years", "max_experience_years"):
            out.append(_experience_evidence(text, f, lo, hi))
        elif f == "seniority" and isinstance(v, str):
            out.append(_seniority_evidence(text, v))
        elif isinstance(v, str):
            out.append(_string_evidence(text, f, v, toks, ALIAS_FIELDS.get(f)))
        else:
            out.append(Evidence(f, v, "unsupported", "none", [], 0.0))
    return out


def missing_cues(text: str, record: dict[str, Any] | None) -> dict[str, Span]:
    """For fields that are null/empty in ``record``: the first text span suggesting they are stated."""
    rec = record or {}
    out: dict[str, Span] = {}

    def empty(f: str) -> bool:
        return rec.get(f) in (None, [], "")

    m = matchers()
    if empty("salary"):
        sal = find_salaries(text)
        if sal:
            out["salary"] = _span(text, sal[0].start, sal[0].end)
    if empty("min_experience_years") and empty("max_experience_years"):
        exp = find_experience(text)
        if exp:
            out["experience"] = _span(text, exp[0].start, exp[0].end)
    for f in ("skills", "locations", "work_mode", "employment_type", "education"):
        if empty(f):
            hits = m[ALIAS_FIELDS[f]].find(text)
            if hits:
                out[f] = _span(text, hits[0].start, hits[0].end)
    return out


def summarise(evidence: list[Evidence]) -> dict[str, Any]:
    """Counts by status, and the share of values that are supported."""
    n = len(evidence)
    by = {s: sum(1 for e in evidence if e.status == s) for s in ("supported", "weak", "unsupported")}
    return {"n_values": n, **by, "supported_share": round(by["supported"] / n, 4) if n else None}


def evidence_by_field(evidence: list[Evidence]) -> dict[str, Status]:
    """Worst status per field (a list field is only as supported as its least supported element)."""
    rank = {"supported": 0, "weak": 1, "unsupported": 2}
    out: dict[str, Status] = {}
    for e in evidence:
        if e.field not in out or rank[e.status] > rank[out[e.field]]:
            out[e.field] = e.status
    return out
