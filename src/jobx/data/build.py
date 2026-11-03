"""End-to-end dataset build: generate -> validate -> split -> dedup -> write -> stats."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from jobx.data.dedup import dedup
from jobx.data.generate import GenConfig, generate
from jobx.data.split import SPLITS, SplitSpec, leakage_report, split_examples
from jobx.data.validate import check_recoverable
from jobx.normalize.salary import to_inr_per_annum
from jobx.schema import FIELD_ORDER


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    return cfg


def _q(vals: Sequence[float], qs: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95)) -> dict[str, float]:
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=float)
    return {f"p{int(q * 100)}": round(float(np.quantile(arr, q)), 2) for q in qs}


def split_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    lab = [r["label"] for r in rows]

    def present(f: str) -> float:
        vals = [x[f] for x in lab]
        return round(sum(1 for v in vals if v not in (None, [], "")) / n, 3)

    mids: list[float] = []
    for x in lab:
        pa = to_inr_per_annum(x["salary"]) if x["salary"] else None
        if pa:
            vals = [v for v in pa if v is not None]
            mids.append(sum(vals) / len(vals) / 1e5)
    return {
        "n": n,
        "families": dict(Counter(r["meta"]["family"] for r in rows).most_common()),
        "domains": dict(Counter(r["meta"].get("domain", "?") for r in rows).most_common()),
        "field_presence": {f: present(f) for f in FIELD_ORDER},
        "seniority": dict(Counter(str(x["seniority"]) for x in lab).most_common()),
        "work_mode": dict(Counter(str(x["work_mode"]) for x in lab).most_common()),
        "employment_type": dict(Counter(str(x["employment_type"]) for x in lab).most_common()),
        "education": dict(Counter(str(x["education"]) for x in lab).most_common()),
        "salary_period": dict(Counter(x["salary"]["period"] for x in lab if x["salary"]).most_common()),
        "salary_currency": dict(Counter(x["salary"]["currency"] for x in lab if x["salary"]).most_common()),
        "salary_open_ended": sum(
            1 for x in lab if x["salary"] and (x["salary"]["min"] is None or x["salary"]["max"] is None)
        ),
        "salary_mid_lpa_quantiles": _q(mids),
        "n_skills_quantiles": _q([len(x["skills"]) for x in lab]),
        "n_locations": dict(Counter(len(x["locations"]) for x in lab).most_common()),
        "top_skills": dict(Counter(s for x in lab for s in x["skills"]).most_common(25)),
        "top_locations": dict(Counter(c for x in lab for c in x["locations"]).most_common(15)),
        "distinct_companies": len({x["company"] for x in lab if x["company"]}),
        "text_chars_quantiles": _q([len(r["text"]) for r in rows]),
        "noise_ops": dict(Counter(o for r in rows for o in r["meta"].get("noise_ops", [])).most_common()),
        "hq_trap_rate": round(sum(1 for r in rows if r["meta"].get("hq_trap")) / n, 3),
        "sources": dict(Counter(r["meta"].get("source", "?") for r in rows).most_common()),
    }


def build(
    config_path: str | Path,
    out_dir: str | Path,
    *,
    paraphrased: str | Path | None = None,
    n_examples: int | None = None,
) -> dict[str, Any]:
    cfg_dict = load_config(config_path)
    if n_examples is not None:
        cfg_dict["n_examples"] = n_examples
    cfg = GenConfig.from_dict(cfg_dict)
    spec = SplitSpec.from_config(cfg_dict)
    examples = generate(cfg)
    n_generated = len(examples)

    val_fail: Counter[str] = Counter()
    valid: list[dict[str, Any]] = []
    for ex in examples:
        res = check_recoverable(ex["text"], ex["label"])
        if res.ok:
            valid.append(ex)
        else:
            val_fail.update(f.split(":")[0] for f in res.failures)

    para_rows: list[dict[str, Any]] = []
    para_stats: dict[str, Any] | None = None
    if paraphrased and Path(paraphrased).exists():
        para_rows = read_jsonl(paraphrased)
        stats_path = Path(paraphrased).with_name("filter_stats.json")
        if stats_path.exists():
            para_stats = json.loads(stats_path.read_text())

    splits, dropped_mismatch = split_examples([*valid, *para_rows], spec)
    flat: list[tuple[str, dict[str, Any]]] = [(s, ex) for s in SPLITS for ex in splits[s]]
    prio_of = {"test_synth": 0, "val": 1, "train": 2}
    prio = [prio_of[s] + (1 if ex["meta"].get("source") == "paraphrase" else 0) for s, ex in flat]
    dd_cfg = cfg_dict.get("dedup", {})
    dres = dedup(
        [ex["text"] for _, ex in flat],
        priority=prio,
        num_perm=int(dd_cfg.get("num_perm", 128)),
        bands=int(dd_cfg.get("bands", 32)),
        shingle_words=int(dd_cfg.get("shingle_words", 3)),
        threshold=float(dd_cfg.get("threshold", 0.8)),
    )
    keep = set(dres.keep)
    cross_split = sum(1 for i, j, _ in dres.near_dups if flat[i][0] != flat[j][0])
    cross_split += sum(1 for i, j in dres.exact_dups if flat[i][0] != flat[j][0])
    final: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    for idx, (s, ex) in enumerate(flat):
        if idx in keep:
            final[s].append(ex)
    for s in SPLITS:
        final[s].sort(key=lambda e: e["id"])

    leak = leakage_report(final)
    if not leak["ok"]:
        raise RuntimeError(f"leakage detected: {leak}")

    out = Path(out_dir)
    for s in SPLITS:
        write_jsonl(out / f"{s}.jsonl", final[s])

    examples_near = [
        {"removed": flat[i][1]["id"], "kept": flat[j][1]["id"], "jaccard": jac}
        for i, j, jac in dres.near_dups[:5]
    ]
    stats: dict[str, Any] = {
        "config": {k: v for k, v in cfg_dict.items() if k not in ("presence",)},
        "presence_config": cfg_dict.get("presence"),
        "pipeline": {
            "generated": n_generated,
            "validation_failed": n_generated - len(valid),
            "validation_failures_by_field": dict(val_fail),
            "paraphrased_added": len(para_rows),
            "dropped_family_company_mismatch": dropped_mismatch,
            "exact_duplicates_removed": len(dres.exact_dups),
            "near_duplicates_removed": len(dres.near_dups),
            "duplicates_removed_across_splits": cross_split,
            "near_duplicate_examples": examples_near,
            "final": {s: len(final[s]) for s in SPLITS},
            "final_total": sum(len(final[s]) for s in SPLITS),
        },
        "paraphrase_filter": para_stats,
        "leakage": leak,
        "splits": {s: split_stats(final[s]) for s in SPLITS},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    return stats
