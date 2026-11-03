"""Leakage-safe split by template family and company.

A test example must share neither its template family nor its company with any train
example; otherwise a model could score well by memorising layouts or company/role
co-occurrences. Assignment is a pure function of (family, company) so the generator,
the paraphrase merge and the final check all agree.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from jobx.normalize.fields import norm_company

SPLITS: tuple[str, ...] = ("train", "val", "test_synth")


@dataclass(frozen=True)
class SplitSpec:
    families: Mapping[str, list[str]]
    company_buckets: Mapping[str, tuple[int, int]]

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> SplitSpec:
        return cls(
            families={k: list(v) for k, v in cfg["families"].items()},
            company_buckets={k: (int(v[0]), int(v[1])) for k, v in cfg["company_buckets"].items()},
        )

    def family_split(self, family: str) -> str:
        base = family.split("/", 1)[-1] if family.startswith("paraphrase/") else family
        for split, fams in self.families.items():
            if base in fams:
                return split
        raise KeyError(f"family {family!r} not assigned to a split")

    def company_split(self, company: str | None) -> str | None:
        key = norm_company(company)
        if key is None:
            return None
        h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 100
        for split, (lo, hi) in self.company_buckets.items():
            if lo <= h < hi:
                return split
        raise ValueError(f"bucket {h} not covered")

    def assign(self, family: str, company: str | None) -> str | None:
        """Split for an example, or None when family and company disagree (drop it)."""
        fs = self.family_split(family)
        cs = self.company_split(company)
        if cs is None or cs == fs:
            return fs
        return None


def split_examples(
    examples: Iterable[dict[str, Any]], spec: SplitSpec
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    dropped = 0
    for ex in examples:
        s = spec.assign(ex["meta"]["family"], ex["label"].get("company"))
        if s is None:
            dropped += 1
            continue
        out[s].append(ex)
    return out, dropped


def leakage_report(splits: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Pairwise overlaps of families and companies between splits (all must be empty)."""
    fams: dict[str, set[str]] = defaultdict(set)
    comps: dict[str, set[str]] = defaultdict(set)
    for s, exs in splits.items():
        for ex in exs:
            f = ex["meta"]["family"]
            fams[s].add(f.split("/", 1)[-1] if f.startswith("paraphrase/") else f)
            c = norm_company(ex["label"].get("company"))
            if c:
                comps[s].add(c)
    names = list(splits)
    rep: dict[str, Any] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            rep[f"{a}~{b}"] = {
                "family_overlap": sorted(fams[a] & fams[b]),
                "company_overlap": sorted(comps[a] & comps[b]),
            }
    rep["ok"] = all(not v["family_overlap"] and not v["company_overlap"] for k, v in rep.items() if k != "ok")
    return rep
