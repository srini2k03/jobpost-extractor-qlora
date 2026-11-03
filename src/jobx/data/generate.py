"""Synthetic postings with labels known by construction.

1. sample a structured record (the gold label) from Indian job-market vocabularies;
2. pick surface forms for each stated field (``surface.py``);
3. lay them out with a template family (``templates.py``);
4. apply noise operators (``noise.py``).
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from jobx import vocab
from jobx.data.noise import apply_noise
from jobx.data.split import SplitSpec
from jobx.data.surface import build_surface
from jobx.data.templates import FAMILIES, SECTIONED, Ctx
from jobx.normalize.text import matchers
from jobx.schema import JobPosting

SENIORITY_WEIGHTS = {"intern": 0.07, "junior": 0.25, "mid": 0.3, "senior": 0.2, "lead": 0.1, "manager": 0.08}


@dataclass
class GenConfig:
    seed: int
    n_examples: int
    block_weights: dict[str, float]
    presence: dict[str, float]
    seniority_in_title: float
    hq_trap: float
    noise: bool
    spec: SplitSpec

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> GenConfig:
        return cls(
            seed=int(cfg["seed"]),
            n_examples=int(cfg["n_examples"]),
            block_weights=dict(cfg["block_weights"]),
            presence=dict(cfg["presence"]),
            seniority_in_title=float(cfg.get("seniority_in_title", 0.7)),
            hq_trap=float(cfg.get("hq_trap", 0.15)),
            noise=bool(cfg.get("noise", True)),
            spec=SplitSpec.from_config(cfg),
        )


def _round_to(x: float, step: float) -> float:
    return round(round(x / step) * step, 2)


def _sample_experience(level: str, rng: random.Random) -> tuple[float | None, float | None]:
    lo_b, hi_b = vocab.SENIORITY_BANDS[level].exp
    if level == "intern":
        return 0.0, 0.0
    if level == "junior" and rng.random() < 0.3:
        return 0.0, 0.0  # freshers
    lo = float(rng.randint(lo_b, max(lo_b, hi_b - 2)))
    r = rng.random()
    if r < 0.12:
        return lo, None  # "5+ years"
    if r < 0.15:
        return None, float(hi_b)  # "upto 2 years"
    hi = float(min(hi_b + 3, lo + rng.choice([1, 2, 2, 3, 3, 4, 5])))
    return lo, hi


def _sample_salary(level: str, tech: bool, remote: bool, rng: random.Random) -> dict[str, Any]:
    band = vocab.SENIORITY_BANDS[level]
    if remote and tech and level in ("mid", "senior", "lead") and rng.random() < 0.08:
        lo = _round_to(rng.uniform(15000, 60000), 5000)
        hi = lo + rng.choice([5000, 10000, 20000])
        return {"min": lo, "max": hi, "currency": "USD", "period": "year"}
    monthly = level == "intern" or (not tech and level == "junior" and rng.random() < 0.5)
    if monthly:
        if level == "intern":
            a, b = band.stipend_month
        else:
            a, b = int(band.nontech_lpa[0] * 1e5 / 12), int(band.nontech_lpa[1] * 1e5 / 12)
        lo = _round_to(rng.uniform(a, b), 1000)
        hi = min(_round_to(lo * rng.uniform(1.1, 1.6), 1000), max(b, lo))
        period = "month"
    else:
        fa, fb = band.tech_lpa if tech else band.nontech_lpa
        step = 0.5 if rng.random() < 0.7 else 1.0
        lo_l = max(fa, _round_to(rng.uniform(fa, fb * 0.8), step))
        hi_l = max(lo_l + step, _round_to(lo_l * rng.uniform(1.15, 1.7), step))
        lo, hi = lo_l * 1e5, hi_l * 1e5
        period = "year"
    lo, hi = round(lo), round(hi)
    r = rng.random()
    out_lo: float | None = lo
    out_hi: float | None = hi
    if r < 0.1:
        out_lo = None  # "upto X"
    elif r < 0.15:
        out_hi = None  # "starting from X"
    elif r < 0.25:
        out_lo = out_hi = rng.choice([lo, hi])  # single figure
    return {"min": out_lo, "max": out_hi, "currency": "INR", "period": period}


def _weighted(rng: random.Random, weights: Mapping[str, float]) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def sample_label(
    rng: random.Random, company: str | None, presence: Mapping[str, float]
) -> tuple[dict[str, Any], str]:
    domain = rng.choices(vocab.DOMAINS, weights=[3 if d.tech else 2 for d in vocab.DOMAINS], k=1)[0]
    role = rng.choice(domain.roles)
    level = _weighted(rng, SENIORITY_WEIGHTS)
    work_mode = _weighted(rng, {"onsite": 0.45, "hybrid": 0.35, "remote": 0.2})
    if level == "intern":
        emp = "internship"
    else:
        emp = _weighted(rng, {"full_time": 0.78, "contract": 0.12, "part_time": 0.06, "freelance": 0.04})

    def present(k: str) -> bool:
        return rng.random() < presence[k]

    skills: list[str] = []
    if present("skills"):
        k = min(len(domain.skills), rng.randint(2, 7))
        skills = rng.sample(domain.skills, k)
    # Skills named in the job title ("React Developer") are stated skills too.
    title_skills = matchers()["skills"].canonicals(role)
    skills = list(dict.fromkeys([*title_skills, *skills]))
    locs: list[str] = []
    if present("locations") and not (work_mode == "remote" and rng.random() < 0.7):
        n = rng.choices([1, 2, 3], weights=[0.7, 0.2, 0.1], k=1)[0]
        cities = list(vocab.CITY_WEIGHTS)
        while len(locs) < n:
            c = rng.choices(cities, weights=[vocab.CITY_WEIGHTS[x] for x in cities], k=1)[0]
            if c not in locs:
                locs.append(c)
    lo, hi = _sample_experience(level, rng) if present("experience") else (None, None)
    label: dict[str, Any] = {
        "company": company if (company is not None and present("company")) else None,
        "role": role,
        "seniority": level if present("seniority") else None,
        "skills": skills,
        "min_experience_years": lo,
        "max_experience_years": hi,
        "salary": _sample_salary(level, domain.tech, work_mode == "remote", rng)
        if present("salary")
        else None,
        "locations": locs,
        "work_mode": work_mode if present("work_mode") else None,
        "employment_type": emp if (level == "intern" or present("employment_type")) else None,
        "education": rng.choice(domain.education) if present("education") else None,
    }
    if label["seniority"] == "intern":
        label["employment_type"] = "internship"  # an "Intern" title states the employment type
    # Round-trip through the schema so every gold label is schema-valid and canonical.
    return JobPosting.model_validate(label).to_json_dict(), domain.name


def companies_by_split(spec: SplitSpec) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {s: [] for s in spec.families}
    for c in vocab.company_names():
        s = spec.company_split(c)
        if s is not None:
            out[s].append(c)
    return out


def generate(cfg: GenConfig) -> list[dict[str, Any]]:
    rng = random.Random(cfg.seed)
    pools = companies_by_split(cfg.spec)
    # Fixed per-split company subsets so companies recur (as in real job boards).
    pools = {
        s: rng.sample(p, min(len(p), {"train": 260, "val": 60, "test_synth": 80}.get(s, 60)))
        for s, p in pools.items()
    }
    examples: list[dict[str, Any]] = []
    for i in range(cfg.n_examples):
        block = _weighted(rng, cfg.block_weights)
        family = rng.choice(cfg.spec.families[block])
        company = rng.choice(pools[block])
        label, domain = sample_label(rng, company, cfg.presence)
        if family == "consultancy_mail" and rng.random() < 0.5:
            label["company"] = None  # "one of our reputed MNC clients"
        hq = None
        if label["locations"] and label["company"] and rng.random() < cfg.hq_trap:
            others = [c for c in vocab.CITY_WEIGHTS if c not in label["locations"]]
            hq = rng.choice(vocab.CITY_ALIASES[rng.choice(others)])
        surface = build_surface(label, rng, seniority_in_title=rng.random() < cfg.seniority_in_title)
        ctx = Ctx(rng=rng, domain=domain, consultancy=rng.choice(vocab.CONSULTANCY_NAMES), hq_city=hq)
        text, variant = FAMILIES[family](surface, ctx)
        ops: list[str] = []
        if cfg.noise:
            text, ops = apply_noise(text, rng, surface.protected, sectioned=family in SECTIONED)
        examples.append(
            {
                "id": f"syn-{i:05d}",
                "text": text,
                "label": label,
                "meta": {
                    "source": "synthetic",
                    "family": family,
                    "template": f"{family}/v{variant}",
                    "domain": domain,
                    "noise_ops": ops,
                    "hq_trap": hq is not None,
                    "block": block,
                },
            }
        )
    return examples
