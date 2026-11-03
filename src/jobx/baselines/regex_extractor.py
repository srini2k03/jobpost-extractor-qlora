"""Rule-based "regex baseline" extractor.

It exists to (1) prove the evaluation stack end to end on CPU and (2) give a floor that a
learned model must beat. Note that it uses the same alias tables as the synthetic generator,
so it is optimistic on synthetic data and expected to be much weaker on real postings.
"""

from __future__ import annotations

import json
import re
from typing import Any

from jobx.normalize.experience import find_experience
from jobx.normalize.fields import norm_role
from jobx.normalize.salary import find_salaries
from jobx.normalize.text import matchers
from jobx.schema.models import FIELD_ORDER
from jobx.vocab import LEGAL_SUFFIXES

_END = r"(?=\s*(?:\||\n|$|\. |,|!|\(|<br|;|•))"
COMPANY_PATTERNS = [
    rf"(?im)^\W*(?:company(?:\s+name)?|co|organi[sz]ation)\s*[:\-|]+\s*(?P<v>[^\n|]+?){_END}",
    rf"(?i)our client\s+(?P<v>[A-Z][\w&.\- ]+?){_END}",
    r"(?m)(?P<v>[A-Z][\w&.\- ]+?)\s+is (?:hiring|a fast-growing|one of|a [a-zA-Z\- ]+ company)",
    rf"(?i)\bat\s+(?P<v>[A-Z][\w&.\- ]+?)(?:\s+in\s|{_END})",
    rf"@\s+(?P<v>[A-Z][\w&.\- ]+?){_END}",
    r"(?i)\bjoin\s+(?P<v>[A-Z][\w&.\- ]+?)\s+as\b",
    r"(?P<v>[A-Z][\w&.\- ]+?)\s+mein\b",
    rf"(?im)^subject:.*\|\s*(?P<v>[^\n|]+?){_END}",
]
ROLE_PATTERNS = [
    rf"(?im)^\W*(?:job\s+title|role|designation|position|opening for)\s*[:\-|]+\s*(?P<v>[^\n|]+?){_END}",
    r"(?i)hiring (?:a|an)\s+(?P<v>.+?)(?:\s+in\s|\s*\(|!|\.|\n)",
    r"(?i)#hiring\s+(?P<v>.+?)\s*(?:@|\||\n)",
    r"(?i)we're hiring!\s+(?P<v>.+?)\s*(?:@|\||\n)",
    r"(?i)walk-in (?:drive for|interview \|)\s+(?P<v>.+?)(?:\s+at\s|\n|$)",
    r"(?i)(?:position of|as an?)\s+(?P<v>.+?)(?:\s+with\s|\.|\n)",
    r"(?P<v>[^\n]+?)\s+ki requirement",
    r"(?i)opportunity:\s+(?P<v>.+?)(?:\s+at\s|\s+in\s|\.|\n)",
    r"(?i)hiring\W+(?P<v>[A-Za-z][^@|\n]+?)\s*(?:@|\||\n|\sat\s)",
]
_SEN_TITLE = [
    ("intern", r"\bintern(ship)?\b"),
    ("manager", r"\bmanager(?:ial| level)\b|\bmanager,"),
    ("senior", r"\b(senior|sr)\b"),
    ("junior", r"\b(junior|jr|trainee|entry[- ]level)\b"),
    ("lead", r"\blead\b"),
    ("mid", r"\bmid[- ]level\b|\bII\b"),
]
_LEGAL_RE = re.compile(
    r"\s*(?:\(OPC\)\s*)?(?:" + "|".join(re.escape(s) for s in LEGAL_SUFFIXES) + r")\s*$", re.IGNORECASE
)
_HQ_RE = re.compile(r"headquartered in ([^.\n]+)", re.IGNORECASE)


def _first(patterns: list[str], text: str) -> str | None:
    for p in patterns:
        m = re.search(p, text)
        if m:
            v = m.group("v").strip(" *:-|\t")
            if v:
                return v
    return None


def _clean_company(v: str | None) -> str | None:
    if not v:
        return None
    v = _LEGAL_RE.sub("", v.strip()).strip(" .,")
    if v.lower() in ("us", "our team", "a leading company", "we") or len(v) > 60:
        return None
    return v or None


def _seniority(title: str | None, text: str) -> str | None:
    if title:
        for level, rx in _SEN_TITLE:
            if re.search(rx, title, re.IGNORECASE if level != "mid" else 0):
                return level
    m = re.search(r"(?im)^\W*(?:level|seniority|career level|grade/level)\s*[:\-]+\s*([^\n|]+)", text)
    if m:
        hits = matchers()["seniority"].canonicals(m.group(1))
        if hits:
            return hits[0]
    return None


def _clean_role(title: str | None) -> str | None:
    if not title:
        return None
    t = re.sub(r"\((?:senior|lead|mid-level|managerial|internship)\)", "", title, flags=re.IGNORECASE)
    t = re.sub(r"^(?:manager,|intern -)\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*-\s*(?:team lead|junior|manager level)\s*$", "", t, flags=re.IGNORECASE)
    t = re.sub(
        r"\b(?:senior|sr\.?|junior|jr\.?|trainee|lead|entry-level|mid-level|intern)\b",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"\bII\b", "", t)
    t = re.sub(r"[*_]+", "", t)
    t = re.sub(r"\s+", " ", t).strip(" -,:.")
    if not t or not norm_role(t):
        return None
    return t.title() if t.isupper() else t


def extract(text: str) -> dict[str, Any]:
    m = matchers()
    title = _first(ROLE_PATTERNS, text)
    if title is None:
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        title = first.strip(" *#🔥🚀") or None
    hq_spans = [mm.span(1) for mm in _HQ_RE.finditer(text)]
    locs = [h.canonical for h in m["locations"].find(text) if not any(a <= h.start < b for a, b in hq_spans)]
    exp = find_experience(text)
    sal = find_salaries(text)

    def first_canon(field: str) -> str | None:
        hits = m[field].canonicals(text)
        return hits[0] if hits else None

    employment = first_canon("employment_type")
    out: dict[str, Any] = {
        "company": _clean_company(_first(COMPANY_PATTERNS, text)),
        "role": _clean_role(title),
        "seniority": _seniority(title, text),
        "skills": m["skills"].canonicals(text),
        "min_experience_years": exp[0].min if exp else None,
        "max_experience_years": exp[0].max if exp else None,
        "salary": sal[0].as_dict() if sal else None,
        "locations": list(dict.fromkeys(locs)),
        "work_mode": first_canon("work_mode"),
        "employment_type": employment,
        "education": first_canon("education"),
    }
    return {k: out[k] for k in FIELD_ORDER}


def predict(text: str) -> str:
    """Model-like interface: returns the JSON string."""
    return json.dumps(extract(text), ensure_ascii=False)
