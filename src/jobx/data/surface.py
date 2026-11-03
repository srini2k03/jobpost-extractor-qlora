"""Render gold field values as the many surface forms Indian postings use.

Each formatter takes a gold value and a ``random.Random`` and returns text whose meaning is
exactly the gold value. The generator relies on this: a label is correct by construction
because the only path from label to text goes through these functions.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any

from jobx import vocab

_WORD = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    12: "twelve",
    15: "fifteen",
}


def _fmt_num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def indian_commas(n: int) -> str:
    s = str(int(n))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts: list[str] = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join([*parts, tail])


def fmt_salary(sal: dict[str, Any], rng: random.Random) -> str:
    lo, hi, cur, period = sal["min"], sal["max"], sal["currency"], sal["period"]
    if cur == "USD":
        a = None if lo is None else f"{_fmt_num(lo / 1000)}k"
        b = None if hi is None else f"{_fmt_num(hi / 1000)}k"
        if a and b and a != b:
            return rng.choice([f"${a}-{b}/yr", f"USD {a} - {b} per year", f"$ {a} to {b} per annum"])
        v = a or b
        pre = "up to " if lo is None else ("min " if hi is None else "")
        return f"{pre}${v}/yr"
    if period == "month":

        def k(x: float) -> str:
            return f"{_fmt_num(x / 1000)}k"

        def rs(x: float) -> str:
            return f"₹{indian_commas(int(x))}"

        if lo is not None and hi is not None and lo != hi:
            forms = [
                f"{rs(lo)} - {rs(hi)} per month",
                f"{k(lo)}-{k(hi)}/month",
                f"Rs {int(lo)}-{int(hi)} pm",
                f"₹{k(lo)} to ₹{k(hi)} per month",
                f"{_fmt_num(lo / 1000)}-{_fmt_num(hi / 1000)}k p.m.",
                f"INR {indian_commas(int(lo))} - {indian_commas(int(hi))} monthly",
            ]
            if lo >= 100000:
                forms.append(f"{_fmt_num(lo / 1e5)}L - {_fmt_num(hi / 1e5)}L per month")
            return rng.choice(forms)
        v = lo if lo is not None else hi
        assert v is not None
        forms = [
            f"{rs(v)}/- per month",
            f"{k(v)}/month",
            f"₹{k(v)} pm",
            f"Rs. {indian_commas(int(v))} per month",
        ]
        if v >= 100000:
            forms.append(f"{_fmt_num(v / 1e5)}L/month")
        s = rng.choice(forms)
        if lo is None:
            return f"{rng.choice(['upto', 'up to', 'Max'])} {s}"
        if hi is None:
            return f"{rng.choice(['Min.', 'starting from', 'from'])} {s}"
        return s

    # annual INR
    def lpa(x: float) -> str:
        return _fmt_num(round(x / 1e5, 2))

    if lo is not None and hi is not None and lo != hi:
        a, b = lpa(lo), lpa(hi)
        return rng.choice(
            [
                f"₹{a}–{b} LPA",
                f"{a}-{b} LPA",
                f"{a} - {b} lpa",
                f"Rs. {a} to {b} Lakhs per annum",
                f"{a}-{b} lakhs",
                f"{a}L - {b}L",
                f"₹ {a}L-{b}L PA",
                f"{a}-{b} Lacs",
                f"INR {indian_commas(int(lo))} - {indian_commas(int(hi))} p.a.",
                f"{indian_commas(int(lo))}-{indian_commas(int(hi))}/yr",
                f"{a} to {b} LPA",
            ]
        )
    v = lo if lo is not None else hi
    assert v is not None
    s = rng.choice(
        [f"{lpa(v)} LPA", f"₹{lpa(v)} LPA", f"{lpa(v)} lakhs per annum", f"INR {indian_commas(int(v))} p.a."]
    )
    if lo is None:
        return f"{rng.choice(['upto', 'up to', 'Max'])} {s}"
    if hi is None:
        return f"{rng.choice(['Min.', 'starting from', 'Minimum'])} {s}"
    return s


def fmt_experience(lo: float | None, hi: float | None, rng: random.Random) -> str:
    yrs = rng.choice(["years", "yrs", "Yrs", "years", "yr"])
    if lo == 0 and hi == 0:
        return rng.choice(["Freshers", "Fresher", "Freshers only", "Freshers welcome"])
    if lo is not None and hi is not None:
        a, b = _fmt_num(lo), _fmt_num(hi)
        if (
            rng.random() < 0.07
            and int(lo) in _WORD
            and int(hi) in _WORD
            and lo.is_integer()
            and hi.is_integer()
        ):
            return f"{_WORD[int(lo)]} to {_WORD[int(hi)]} years"
        return rng.choice([f"{a}-{b} {yrs}", f"{a} - {b} {yrs}", f"{a} to {b} {yrs}", f"{a}–{b} {yrs}"])
    if lo is not None:
        a = _fmt_num(lo)
        return rng.choice([f"{a}+ {yrs}", f"minimum {a} {yrs}", f"Min {a} {yrs}", f"at least {a} {yrs}"])
    assert hi is not None
    return f"{rng.choice(['upto', 'up to', 'max'])} {_fmt_num(hi)} {yrs}"


def fmt_locations(locs: list[str], rng: random.Random) -> str:
    names = [rng.choice(vocab.CITY_ALIASES[c]) for c in locs]
    if len(names) == 1:
        return names[0]
    sep = rng.choice([", ", " / ", "/", " | ", " & "])
    if sep == " & " and len(names) > 2:
        return ", ".join(names[:-1]) + " & " + names[-1]
    return sep.join(names)


def fmt_skills(skills: list[str], rng: random.Random) -> list[str]:
    return [rng.choice(vocab.SKILL_ALIASES[s]) for s in skills]


def fmt_title(role: str, seniority: str | None, rng: random.Random) -> str:
    if seniority is None:
        return role
    return rng.choice(vocab.SENIORITY_TITLE_FORMS[seniority]).format(role=role)


def fmt_enum(aliases: dict[str, list[str]], value: str, rng: random.Random) -> str:
    return rng.choice(aliases[value])


@dataclass
class Surface:
    """Surface strings for one record. ``None`` means the field is absent from the text."""

    company: str | None
    title: str
    seniority_line: str | None
    skills: list[str]
    experience: str | None
    salary: str | None
    locations: str | None
    work_mode: str | None
    employment_type: str | None
    education: str | None
    protected: set[str] = field(default_factory=set)


def build_surface(label: dict[str, Any], rng: random.Random, *, seniority_in_title: bool) -> Surface:
    company = label["company"]
    if company is not None and rng.random() < 0.35:
        company = f"{company} {rng.choice(vocab.LEGAL_SUFFIXES)}"
    sen = label["seniority"]
    title = fmt_title(label["role"], sen if seniority_in_title else None, rng) if label["role"] else ""
    sen_line = None
    if sen is not None and not seniority_in_title:
        sen_line = rng.choice(vocab.SENIORITY_LINE_FORMS[sen])
    exp = None
    if label["min_experience_years"] is not None or label["max_experience_years"] is not None:
        exp = fmt_experience(label["min_experience_years"], label["max_experience_years"], rng)
    sal = fmt_salary(label["salary"], rng) if label["salary"] else None
    locs = fmt_locations(label["locations"], rng) if label["locations"] else None
    wm = fmt_enum(vocab.WORK_MODE_ALIASES, label["work_mode"], rng) if label["work_mode"] else None
    et = None
    if label["employment_type"]:
        # "Intern" as an employment-type alias reads oddly next to an intern title.
        et = fmt_enum(vocab.EMPLOYMENT_TYPE_ALIASES, label["employment_type"], rng)
    edu = fmt_enum(vocab.EDUCATION_ALIASES, label["education"], rng) if label["education"] else None
    s = Surface(company, title, sen_line, fmt_skills(label["skills"], rng), exp, sal, locs, wm, et, edu)
    words: set[str] = set()
    for part in [company, title, sen_line, exp, sal, locs, wm, et, edu, *s.skills]:
        if part:
            words.update(w.lower() for w in re.findall(r"[A-Za-z]+", part))
    s.protected = words
    return s
