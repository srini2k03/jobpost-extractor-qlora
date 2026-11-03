"""Field-recoverability checker for synthetic and paraphrased postings.

* :func:`check_recoverable` - every gold value can be found in the text with the same rules
  the normalisers use (aliases, salary and experience parsers).
* :func:`check_no_new_facts` - a paraphrase must not introduce field values (cities, skills,
  salaries, experience ranges, work modes...) that were absent from the source text.
* An optional NLI hook adds an entailment check per field, for the cases rules can't see.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from jobx.normalize.experience import find_experience
from jobx.normalize.fields import norm_company, norm_role
from jobx.normalize.salary import find_salaries, salary_key
from jobx.normalize.text import loose, matchers

NLIScorer = Callable[[str, str], float]
"""(premise, hypothesis) -> probability of entailment."""

_SENIORITY_CUES: dict[str, str] = {
    "intern": r"\bintern(ship)?\b",
    "junior": r"\b(junior|jr|trainee|entry)\b",
    "mid": r"\b(mid|ii)\b",
    "senior": r"\b(senior|sr)\b",
    "lead": r"\blead\b",
    "manager": r"\bmanager(ial)?\b",
}


@dataclass
class CheckResult:
    ok: bool
    failures: list[str] = field(default_factory=list)


def _salary_keys(text: str) -> set[Any]:
    return {salary_key(m.as_dict()) for m in find_salaries(text)}


def _exp_keys(text: str) -> set[tuple[float | None, float | None]]:
    return {(m.min, m.max) for m in find_experience(text)}


def check_recoverable(text: str, label: dict[str, Any]) -> CheckResult:
    fails: list[str] = []
    lt = loose(text)
    m = matchers()
    if label.get("company"):
        key = norm_company(label["company"])
        if key and key not in lt:
            fails.append("company")
    if label.get("role"):
        key = norm_role(label["role"])
        if key and loose(label["role"]) not in lt and key not in (norm_role(text) or ""):
            fails.append("role")
    if label.get("seniority"):
        cue = _SENIORITY_CUES[label["seniority"]]
        if not re.search(cue, lt):
            fails.append("seniority")
    found_skills = set(m["skills"].canonicals(text))
    fails += [f"skills:{s}" for s in label.get("skills") or [] if s not in found_skills]
    found_locs = set(m["locations"].canonicals(text))
    fails += [f"locations:{c}" for c in label.get("locations") or [] if c not in found_locs]
    for f in ("work_mode", "employment_type", "education"):
        if label.get(f) and label[f] not in m[f].canonicals(text):
            fails.append(f)
    lo, hi = label.get("min_experience_years"), label.get("max_experience_years")
    if lo is not None or hi is not None:
        want = (None if lo is None else float(lo), None if hi is None else float(hi))
        if want not in _exp_keys(text):
            fails.append("experience")
    if label.get("salary") and salary_key(label["salary"]) not in _salary_keys(text):
        fails.append("salary")
    return CheckResult(not fails, fails)


def extracted_facts(text: str) -> dict[str, set[Any]]:
    m = matchers()
    return {
        "skills": set(m["skills"].canonicals(text)),
        "locations": set(m["locations"].canonicals(text)),
        "work_mode": set(m["work_mode"].canonicals(text)),
        "employment_type": set(m["employment_type"].canonicals(text)),
        "education": set(m["education"].canonicals(text)),
        "salary": _salary_keys(text),
        "experience": _exp_keys(text),
    }


def check_no_new_facts(source: str, paraphrase: str) -> CheckResult:
    src, par = extracted_facts(source), extracted_facts(paraphrase)
    fails = [f"new:{k}:{sorted(map(str, par[k] - src[k]))}" for k in par if par[k] - src[k]]
    return CheckResult(not fails, fails)


def hypotheses(label: dict[str, Any]) -> list[tuple[str, str]]:
    """Natural-language statements implied by the gold label, for an NLI model."""
    h: list[tuple[str, str]] = []
    if label.get("company"):
        h.append(("company", f"The hiring company is {label['company']}."))
    if label.get("role"):
        h.append(("role", f"The job is for a {label['role']}."))
    if label.get("locations"):
        h.append(("locations", f"The job is located in {', '.join(label['locations'])}."))
    sal = label.get("salary")
    if sal:
        unit = "per month" if sal.get("period") == "month" else "per year"
        if sal.get("min") is not None and sal.get("max") is not None:
            h.append(
                (
                    "salary",
                    f"The salary is between {sal['min']:,} and {sal['max']:,} {sal['currency']} {unit}.",
                )
            )
        else:
            v = sal.get("min") if sal.get("min") is not None else sal.get("max")
            h.append(("salary", f"The salary is about {v:,} {sal['currency']} {unit}."))
    lo, hi = label.get("min_experience_years"), label.get("max_experience_years")
    if lo == 0 and hi == 0:
        h.append(("experience", "Freshers can apply for this job."))
    elif lo is not None and hi is not None:
        h.append(("experience", f"The job requires {lo} to {hi} years of experience."))
    if label.get("work_mode"):
        h.append(
            (
                "work_mode",
                {
                    "onsite": "The job is in the office.",
                    "hybrid": "The job is hybrid.",
                    "remote": "The job is remote.",
                }[label["work_mode"]],
            )
        )
    return h


def validate_paraphrase(
    source: str,
    paraphrase: str,
    label: dict[str, Any],
    *,
    nli: NLIScorer | None = None,
    nli_threshold: float = 0.5,
) -> CheckResult:
    """Rule checks (recoverable + nothing new) and, if given, NLI entailment per field."""
    fails = check_recoverable(paraphrase, label).failures + check_no_new_facts(source, paraphrase).failures
    if nli is not None:
        for fname, hyp in hypotheses(label):
            if nli(paraphrase, hyp) < nli_threshold:
                fails.append(f"nli:{fname}")
    return CheckResult(not fails, fails)


def hf_nli_scorer(
    model_name: str = "cross-encoder/nli-deberta-v3-small", device: int | str = -1
) -> NLIScorer:
    """NLI scorer backed by a Hugging Face cross-encoder (imported lazily; used in Colab)."""
    from transformers import pipeline

    clf = pipeline("text-classification", model=model_name, device=device, top_k=None)

    def score(premise: str, hypothesis: str) -> float:
        out = clf({"text": premise[:2000], "text_pair": hypothesis})
        rows = out[0] if isinstance(out, list) and out and isinstance(out[0], list) else out
        for r in rows:
            if str(r["label"]).lower().startswith("entail"):
                return float(r["score"])
        return 0.0

    return score
