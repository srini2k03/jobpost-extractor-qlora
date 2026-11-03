"""Per-field normalisation used for scoring (applied identically to gold and prediction)."""

from __future__ import annotations

import re
from typing import Any

from jobx.normalize.salary import salary_key
from jobx.normalize.text import loose, matchers
from jobx.vocab import LEGAL_SUFFIXES, SENIORITY_WORDS

_LEGAL_RE = re.compile(
    r"\b("
    + "|".join(
        sorted(
            {loose(s) for s in LEGAL_SUFFIXES}
            | {"pvt", "ltd", "private", "limited", "opc", "inc", "llp", "pvt ltd"},
            key=len,
            reverse=True,
        )
    )
    + r")\b"
)
_SEN_WORDS = sorted({loose(w) for w in SENIORITY_WORDS}, key=len, reverse=True)
_SEN_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in _SEN_WORDS) + r")\b")

WORD_NUMBERS = {
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
    "eleven": 11,
    "twelve": 12,
    "fifteen": 15,
}


def norm_company(v: Any) -> str | None:
    if not isinstance(v, str) or not v.strip():
        return None
    s = loose(v)
    s = _LEGAL_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def norm_role(v: Any) -> str | None:
    if not isinstance(v, str) or not v.strip():
        return None
    s = loose(v.replace("/", " / "))
    s = _SEN_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _norm_enum(field: str, v: Any) -> str | None:
    if not isinstance(v, str) or not v.strip():
        return None
    s = v.strip().lower()
    canon = matchers()[field].canonicalise(s)
    if canon is not None:
        return canon
    return s.replace("-", "_").replace(" ", "_")


def norm_number(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int | float):
        return round(float(v), 1)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in WORD_NUMBERS:
            return float(WORD_NUMBERS[s])
        m = re.search(r"\d+(?:\.\d+)?", s)
        return round(float(m.group()), 1) if m else None
    return None


def norm_list(field: str, v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        v = [x for x in re.split(r"[,;/|]", v) if x.strip()]
    if not isinstance(v, list):
        return []
    out: dict[str, None] = {}
    m = matchers()[field]
    for x in v:
        if not isinstance(x, str) or not x.strip():
            continue
        canon = m.canonicalise(x)
        out.setdefault(canon if canon is not None else loose(x), None)
    return list(out)


def normalise_field(field: str, v: Any) -> Any:
    """Return a hashable, comparable value; None (or empty tuple for lists) means absent."""
    if field == "company":
        return norm_company(v)
    if field == "role":
        return norm_role(v)
    if field in ("seniority", "work_mode", "employment_type", "education"):
        return _norm_enum(field, v)
    if field in ("min_experience_years", "max_experience_years"):
        return norm_number(v)
    if field == "salary":
        return salary_key(v)
    if field in ("skills", "locations"):
        return tuple(sorted(norm_list(field, v)))
    raise KeyError(field)


def normalise_record(d: dict[str, Any] | None) -> dict[str, Any]:
    from jobx.schema.models import FIELD_ORDER

    d = d or {}
    return {f: normalise_field(f, d.get(f)) for f in FIELD_ORDER}


def canonical_skills(skills: list[str]) -> list[str]:
    return norm_list("skills", skills)


def canonical_locations(locs: list[str]) -> list[str]:
    return norm_list("locations", locs)
