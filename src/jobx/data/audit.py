"""Independent leakage / memorisation audit of the committed splits.

The build (``jobx.data.build``) already removes duplicates at Jaccard >= 0.8 and asserts that template
families and companies are disjoint. This audit re-checks the *committed files* with stricter and
complementary tests, so a regression in the build or a hand-edited split cannot slip through:

* exact duplicates of the raw and of the normalised text, within and across splits;
* near duplicates across splits: MinHash (128 permutations) estimates for every cross-split pair, then
  exact word-3-gram Jaccard for every pair whose estimate is >= ``screen``; counts at 0.5 / 0.6 / 0.7 / 0.8;
* nearest-train similarity of every eval example (quantiles of the exact Jaccard to its closest train item);
* n-gram contamination: share of each eval example's word 8-grams that also occur anywhere in train;
* label overlap: identical full gold records and identical (role, skills) pairs across splits;
* split metadata: template families, template ids and companies must be disjoint;
* few-shot exemplars used by the prompting baselines must come from train.

``jobx audit`` writes ``results/audit/leakage.json``; ``tests/test_audit.py`` enforces the thresholds in
``GATES`` on the committed data, so CI fails if leakage appears.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from jobx.data.dedup import MinHasher, jaccard, shingles
from jobx.normalize.fields import norm_company
from jobx.normalize.text import loose

SPLIT_ORDER: tuple[str, ...] = ("train", "val", "test_synth", "test_real")
THRESHOLDS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8)
NGRAM = 8

# Hard limits enforced by tests/test_audit.py on the committed data.
GATES: dict[str, float] = {
    "max_cross_split_exact_duplicates": 0,
    "max_cross_split_pairs_jaccard_ge_0.6": 0,
    "max_eval_mean_8gram_overlap_with_train": 0.10,
    "max_cross_split_identical_records": 0,
}


def _sha(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def _quant(vals: Sequence[float]) -> dict[str, float]:
    if not len(vals):
        return {}
    a = np.asarray(vals, dtype=float)
    out = {f"p{q}": round(float(np.percentile(a, q)), 4) for q in (50, 90, 95, 99)}
    out["max"] = round(float(a.max()), 4)
    out["mean"] = round(float(a.mean()), 4)
    return out


def ngrams(text: str, n: int = NGRAM) -> set[int]:
    toks = re.findall(r"[a-z0-9+#]+", loose(text))
    return {hash(" ".join(toks[i : i + n])) for i in range(len(toks) - n + 1)}


def _record_key(label: Mapping[str, Any]) -> str:
    return json.dumps(label, sort_keys=True, ensure_ascii=False)


def _role_skills_key(label: Mapping[str, Any]) -> str:
    return json.dumps([loose(str(label.get("role") or "")), sorted(label.get("skills") or [])])


def cross_split_near_duplicates(
    texts: Mapping[str, list[str]], *, screen: float = 0.4, num_perm: int = 128, shingle_words: int = 3
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    """All cross-split pairs with exact Jaccard >= ``screen`` (screened by MinHash estimates), and for
    every non-train example its exact Jaccard to the most similar train example."""
    mh = MinHasher(num_perm)
    sh = {s: [shingles(t, shingle_words) for t in ts] for s, ts in texts.items()}
    sig = {s: np.stack([mh.signature(x) for x in v]) if v else np.zeros((0, num_perm)) for s, v in sh.items()}
    names = [s for s in SPLIT_ORDER if s in texts]
    pairs: list[dict[str, Any]] = []
    nearest: dict[str, list[float]] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            sa, sb = sig[a], sig[b]
            for r0 in range(0, len(sa), 64):
                est = (sa[r0 : r0 + 64, None, :] == sb[None, :, :]).mean(axis=2)
                for ii, jj in zip(*np.nonzero(est >= screen - 0.15), strict=True):
                    jac = jaccard(sh[a][r0 + ii], sh[b][jj])
                    if jac >= screen:
                        pairs.append(
                            {"a": a, "i": int(r0 + ii), "b": b, "j": int(jj), "jaccard": round(jac, 4)}
                        )
    if "train" in sig:
        st = sig["train"]
        for s in names:
            if s == "train":
                continue
            best: list[float] = []
            for k, row in enumerate(sig[s]):
                est = (st == row[None, :]).mean(axis=1)
                top = np.argsort(-est)[:5]  # exact Jaccard on the 5 best MinHash candidates
                best.append(max(jaccard(sh[s][k], sh["train"][int(j)]) for j in top) if len(top) else 0.0)
            nearest[s] = best
    return pairs, nearest


def audit(
    splits: Mapping[str, list[dict[str, Any]]], *, few_shot_ids: Sequence[str] = (), screen: float = 0.4
) -> dict[str, Any]:
    names = [s for s in SPLIT_ORDER if splits.get(s)]
    texts = {s: [r["text"] for r in splits[s]] for s in names}
    ids = {s: [r["id"] for r in splits[s]] for s in names}
    rep: dict[str, Any] = {"n": {s: len(texts[s]) for s in names}}

    # exact duplicates (raw text and normalised text)
    exact: dict[str, Any] = {}
    for kind, norm in (("raw", lambda t: t), ("normalised", loose)):
        where: dict[str, set[str]] = {}
        within: Counter[str] = Counter()
        for s in names:
            seen: set[str] = set()
            for t in texts[s]:
                h = _sha(norm(t))
                if h in seen:
                    within[s] += 1
                seen.add(h)
                where.setdefault(h, set()).add(s)
        exact[kind] = {
            "within_split": {s: within[s] for s in names},
            "cross_split": sum(1 for v in where.values() if len(v) > 1),
        }
    rep["exact_duplicates"] = exact

    # near duplicates across splits
    pairs, nearest = cross_split_near_duplicates(texts, screen=screen)
    rep["near_duplicates"] = {
        "method": f"MinHash(128) screen, exact word-3-gram Jaccard confirmation; pairs >= {screen} listed",
        "cross_split_pairs_at": {
            f"{t}": sum(1 for p in pairs if p["jaccard"] >= t) for t in (screen, *THRESHOLDS)
        },
        "examples": [
            {**p, "a_id": ids[p["a"]][p["i"]], "b_id": ids[p["b"]][p["j"]]}
            for p in sorted(pairs, key=lambda p: -p["jaccard"])[:10]
        ],
        "nearest_train_jaccard": {s: _quant(v) for s, v in nearest.items()},
    }

    # n-gram contamination against train
    if "train" in texts:
        train_ng: set[int] = set().union(*(ngrams(t) for t in texts["train"]))
        contamination: dict[str, Any] = {}
        for s in names:
            if s == "train":
                continue
            fr = []
            for t in texts[s]:
                g = ngrams(t)
                fr.append(len(g & train_ng) / len(g) if g else 0.0)
            contamination[s] = {
                **_quant(fr),
                "share_over_0.5": round(float(np.mean([f > 0.5 for f in fr])), 4) if fr else 0.0,
            }
        rep["ngram_overlap_with_train"] = {"n": NGRAM, **contamination}

    # label overlap
    lab = {s: [r.get("label") or {} for r in splits[s]] for s in names}
    rec = {s: {_record_key(x) for x in lab[s]} for s in names}
    rs = {s: {_role_skills_key(x) for x in lab[s]} for s in names}
    rep["label_overlap"] = {
        f"{a}~{b}": {
            "identical_records": len(rec[a] & rec[b]),
            "identical_role_and_skills": len(rs[a] & rs[b]),
        }
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }

    # split metadata (families, template ids, companies)
    def meta_set(s: str, key: str) -> set[str]:
        out = set()
        for r in splits[s]:
            v = (r.get("meta") or {}).get(key)
            if v:
                out.add(str(v).split("/", 1)[-1] if str(v).startswith("paraphrase/") else str(v))
        return out

    comp = {s: {c for c in (norm_company((x or {}).get("company")) for x in lab[s]) if c} for s in names}
    synth = [s for s in names if s != "test_real"]
    rep["metadata_overlap"] = {
        f"{a}~{b}": {
            "families": sorted(meta_set(a, "family") & meta_set(b, "family")),
            "templates": sorted(meta_set(a, "template") & meta_set(b, "template")),
            "companies": sorted(comp[a] & comp[b])[:20],
            "n_companies": len(comp[a] & comp[b]),
        }
        for i, a in enumerate(synth)
        for b in synth[i + 1 :]
    }

    # few-shot exemplars
    if few_shot_ids:
        where_ids = {i: s for s in names for i in ids[s]}
        rep["few_shot_exemplars"] = {i: where_ids.get(i, "missing") for i in few_shot_ids}

    # gates
    worst_overlap = max(
        (
            v.get("mean", 0.0)
            for k, v in rep.get("ngram_overlap_with_train", {}).items()
            if isinstance(v, dict)
        ),
        default=0.0,
    )
    observed = {
        "max_cross_split_exact_duplicates": exact["normalised"]["cross_split"],
        "max_cross_split_pairs_jaccard_ge_0.6": rep["near_duplicates"]["cross_split_pairs_at"]["0.6"],
        "max_eval_mean_8gram_overlap_with_train": worst_overlap,
        "max_cross_split_identical_records": sum(
            v["identical_records"] for v in rep["label_overlap"].values()
        ),
    }
    rep["gates"] = {
        k: {"limit": GATES[k], "observed": observed[k], "ok": observed[k] <= GATES[k]} for k in GATES
    }
    rep["gates_ok"] = all(v["ok"] for v in rep["gates"].values())
    meta_ok = all(
        not v["families"] and not v["templates"] and not v["n_companies"]
        for v in rep["metadata_overlap"].values()
    )
    few_ok = all(v == "train" for v in rep.get("few_shot_exemplars", {}).values())
    rep["metadata_ok"] = meta_ok
    rep["few_shot_ok"] = few_ok
    rep["ok"] = rep["gates_ok"] and meta_ok and few_ok
    return rep


def few_shot_ids_from_predictions(pred_dir: str | Path) -> list[str]:
    out: dict[str, None] = {}
    for p in sorted(Path(pred_dir).glob("*.meta.json")):
        for i in json.loads(p.read_text()).get("shot_ids") or []:
            out.setdefault(str(i), None)
    return list(out)


def run_audit(
    root: str | Path = ".", out: str | Path | None = "results/audit/leakage.json"
) -> dict[str, Any]:
    from jobx.data.build import read_jsonl
    from jobx.eval.run import SPLIT_FILES

    root = Path(root)
    splits = {s: read_jsonl(root / f) for s, f in SPLIT_FILES.items() if (root / f).exists()}
    rep = audit(splits, few_shot_ids=few_shot_ids_from_predictions(root / "results" / "predictions"))
    if out is not None:
        dest = root / out
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
    return rep
