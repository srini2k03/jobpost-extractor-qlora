"""Locate predicted field values in the posting text (for the Space's highlight view)."""

from __future__ import annotations

import re
from typing import Any

from jobx.normalize.experience import find_experience
from jobx.normalize.fields import norm_number
from jobx.normalize.salary import find_salaries, salary_key
from jobx.normalize.text import matchers


def spans(text: str, pred: dict[str, Any]) -> list[tuple[int, int, str]]:
    found: list[tuple[int, int, str]] = []
    m = matchers()
    skills = set(pred.get("skills") or [])
    locs = set(pred.get("locations") or [])
    found += [(h.start, h.end, "skill") for h in m["skills"].find(text) if h.canonical in skills]
    found += [(h.start, h.end, "location") for h in m["locations"].find(text) if h.canonical in locs]
    for f in ("work_mode", "employment_type", "education"):
        if pred.get(f):
            found += [(h.start, h.end, f) for h in m[f].find(text) if h.canonical == pred[f]][:1]
    if pred.get("salary"):
        key = salary_key(pred["salary"])
        found += [(s.start, s.end, "salary") for s in find_salaries(text) if salary_key(s.as_dict()) == key][
            :1
        ]
    lo, hi = norm_number(pred.get("min_experience_years")), norm_number(pred.get("max_experience_years"))
    if lo is not None or hi is not None:
        found += [
            (e.start, e.end, "experience") for e in find_experience(text) if (e.min, e.max) == (lo, hi)
        ][:1]
    for f in ("company", "role"):
        v = pred.get(f)
        if isinstance(v, str) and v.strip():
            mm = re.search(re.escape(v.strip()), text, re.IGNORECASE)
            if mm:
                found.append((mm.start(), mm.end(), f))
    found.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    out: list[tuple[int, int, str]] = []
    for s, e, lab in found:
        if out and s < out[-1][1]:
            continue
        out.append((s, e, lab))
    return out


def highlighted(text: str, pred: dict[str, Any] | None) -> list[tuple[str, str | None]]:
    """Gradio HighlightedText format: [(segment, label or None), ...]."""
    if not pred:
        return [(text, None)]
    segs: list[tuple[str, str | None]] = []
    pos = 0
    for s, e, lab in spans(text, pred):
        if s > pos:
            segs.append((text[pos:s], None))
        segs.append((text[s:e], lab))
        pos = e
    if pos < len(text):
        segs.append((text[pos:], None))
    return segs
