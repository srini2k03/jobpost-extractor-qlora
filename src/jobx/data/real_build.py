"""Build the real-postings test split from the fetched pools and the annotation file.

    jobx real-build            # pools + data/real/annotations.jsonl -> test_real.jsonl, rejected.json, real_meta.json
    jobx verify-real           # human pass over the annotations (accept / edit / reject), saved after each posting

``data/real/annotations.jsonl`` holds one record per annotated pool posting (format: docs/annotation.md):

    {"id": ..., "label": {...JobPosting fields, omitted = null/[]...},
     "alt": {field: [other acceptable values]}, "notes": "...",
     "human_verified": false}            # or {"id": ..., "reject": "<reason>"}

The labels were produced with AI assistance and are **not human-verified** unless ``human_verified`` is
true (set only by ``jobx verify-real``). Every output row says so in ``annotation``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rich.console import Console

from jobx.data.build import read_jsonl, write_jsonl
from jobx.schema import FIELD_ORDER, JobPosting

POOLS: tuple[str, ...] = ("data/real/pool_hf.jsonl", "data/real/pool_web.jsonl")
ANNOTATIONS = "data/real/annotations.jsonl"
GUIDELINES = "docs/annotation.md"
AI_METHOD = "AI-assisted annotation, not human-verified"


def load_pool(root: Path, pools: Sequence[str] = POOLS) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in pools:
        if (root / p).exists():
            for r in read_jsonl(root / p):
                out[r["id"]] = r
    return out


def validate_annotation(a: dict[str, Any]) -> dict[str, Any]:
    """Canonical label (all keys, schema-validated) and validated alternatives."""
    if "reject" in a:
        return a
    label = JobPosting.model_validate(a["label"]).to_json_dict()
    alts: dict[str, list[Any]] = {}
    for f, vals in (a.get("alt") or {}).items():
        if f not in FIELD_ORDER:
            raise ValueError(f"{a['id']}: unknown field {f!r} in alt")
        alts[f] = [JobPosting.model_validate({**label, f: v}).to_json_dict()[f] for v in vals]
    return {**a, "label": label, "alt": alts}


def build_real(root: str | Path = ".", out_dir: str | Path = "data/real") -> dict[str, Any]:
    root, out = Path(root), Path(out_dir)
    pool = load_pool(root)
    anns = [validate_annotation(a) for a in read_jsonl(root / ANNOTATIONS)]
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for a in anns:
        src = pool.get(a["id"])
        if src is None:
            raise KeyError(f"annotation {a['id']} has no posting in the pools")
        if "reject" in a:
            rejected.append({"id": a["id"], "reason": a["reject"]})
            continue
        verified = bool(a.get("human_verified"))
        rows.append(
            {
                "id": a["id"],
                "text": src["text"],
                "label": a["label"],
                "label_alternatives": a["alt"],
                "source": src["source"],
                "annotation": {
                    "method": "human-verified" if verified else AI_METHOD,
                    "human_verified": verified,
                    "verified_by": a.get("verified_by"),
                    "verified_at": a.get("verified_at"),
                    "guidelines": GUIDELINES,
                    "notes": a.get("notes"),
                },
                "meta": {"source": "real", "family": f"real/{_kind(src)}"},
            }
        )
    rows.sort(key=lambda r: r["id"])
    write_jsonl(out / "test_real.jsonl", rows)
    (out / "rejected.json").write_text(json.dumps(sorted(rejected, key=lambda r: r["id"]), indent=2) + "\n")
    per_source = Counter(_kind(r) for r in rows)
    licences: dict[str, dict[str, Any]] = {}
    for r in rows:
        s = r["source"]
        key = s.get("dataset") or s.get("corpus") or "?"
        lic = licences.setdefault(
            key,
            {
                "licence": s.get("licence") or s.get("corpus_licence"),
                "revision": s.get("revision") or s.get("corpus_revision"),
                "n": 0,
            },
        )
        lic["n"] += 1
    meta = {
        "n": len(rows),
        "rejected": len(rejected),
        "per_source": dict(per_source.most_common()),
        "sources": licences,
        "annotation": {
            "method": AI_METHOD,
            "human_verified": sum(1 for r in rows if r["annotation"]["human_verified"]),
            "with_alternatives": sum(1 for r in rows if r["label_alternatives"]),
            "guidelines": GUIDELINES,
        },
        "chars": _quantiles([len(r["text"]) for r in rows]),
    }
    (out / "real_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _kind(row: dict[str, Any]) -> str:
    s = row.get("source") or {}
    return str(s.get("composition") or ("web" if s.get("corpus") else "unknown"))


def _quantiles(vals: list[int]) -> dict[str, float]:
    if not vals:
        return {}
    v = sorted(vals)
    return {f"p{q}": float(v[min(len(v) - 1, int(q / 100 * len(v)))]) for q in (5, 50, 95)}


# ------------------------------------------------------------------------------------------------------------
# Human verification


def verify_annotations(
    root: str | Path = ".",
    *,
    input_fn: Callable[[str], str] = input,
    console: Console | None = None,
    verifier: str = "human",
    preview_chars: int = 2500,
) -> dict[str, int]:
    """Walk the AI-assisted annotations that are not human-verified yet; accept, edit or reject each.
    The annotation file is rewritten after every decision, so the session can stop at any time."""
    from jobx.verify import _table

    root = Path(root)
    con = console or Console()
    path = root / ANNOTATIONS
    anns = read_jsonl(path)
    pool = load_pool(root)
    stats = {"accepted": 0, "edited": 0, "rejected": 0}
    todo = [i for i, a in enumerate(anns) if "label" in a and not a.get("human_verified")]
    con.print(f"[bold]{len(anns) - len(todo)} done or rejected, {len(todo)} to verify[/bold]")
    for k, i in enumerate(todo):
        a = validate_annotation(anns[i])
        text = pool[a["id"]]["text"]
        label = dict(a["label"])
        edited: list[str] = []
        con.rule(f"{a['id']}  ({k + 1}/{len(todo)})")
        con.print(
            text[:preview_chars] + (" [dim]...(v to view all)[/dim]" if len(text) > preview_chars else "")
        )
        if a.get("alt"):
            con.print(f"[dim]also accepted: {json.dumps(a['alt'], ensure_ascii=False)}[/dim]")
        if a.get("notes"):
            con.print(f"[dim]notes: {a['notes']}[/dim]")
        while True:
            con.print(_table(label))
            cmd = input_fn("[a]ccept / [e]dit / [r]eject / [v]iew / [q]uit > ").strip().lower()
            if cmd == "v":
                con.print(text)
            elif cmd == "q":
                return stats
            elif cmd == "r":
                anns[i] = {"id": a["id"], "reject": input_fn("reason > ").strip() or "rejected by verifier"}
                stats["rejected"] += 1
                break
            elif cmd == "e":
                field = input_fn("field > ").strip()
                if field not in FIELD_ORDER:
                    con.print(f"[red]unknown field {field!r}[/red]")
                    continue
                try:
                    label = JobPosting.model_validate(
                        {**label, field: json.loads(input_fn(f"{field} (JSON) > "))}
                    ).to_json_dict()
                    edited.append(field)
                except (json.JSONDecodeError, ValidationError) as e:
                    con.print(f"[red]rejected: {e}[/red]")
            elif cmd == "a":
                anns[i] = {
                    **anns[i],
                    "label": label,
                    "human_verified": True,
                    "verified_by": verifier,
                    "verified_at": datetime.now(UTC).strftime("%Y-%m-%d"),
                    "edited_fields": sorted(set(edited)),
                }
                stats["edited" if edited else "accepted"] += 1
                break
        write_jsonl(path, anns)
    return stats
