"""Human verification of draft labels for the real test set.

    jobx verify data/real/test_real_draft.jsonl --out data/real/test_real.jsonl

For each posting you see the text and the draft label field by field, then:
  a  accept (as edited so far)       e  edit a field (value typed as JSON, e.g. ["python","sql"] or null)
  s  skip / exclude this posting     v  view full text           q  save and quit
Progress is appended to the output file after every posting, so you can stop and resume.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from jobx.data.build import read_jsonl
from jobx.schema import FIELD_ORDER, JobPosting

InputFn = Callable[[str], str]


def _table(label: dict[str, Any]) -> Table:
    t = Table(show_header=True, header_style="bold")
    t.add_column("field")
    t.add_column("draft value")
    for f in FIELD_ORDER:
        t.add_row(f, json.dumps(label.get(f), ensure_ascii=False))
    return t


def _blank() -> dict[str, Any]:
    return JobPosting().to_json_dict()


def verify(
    draft_path: str | Path,
    out_path: str | Path,
    *,
    input_fn: InputFn = input,
    console: Console | None = None,
    preview_chars: int = 1500,
    verifier: str = "human",
) -> dict[str, Any]:
    con = console or Console()
    drafts = read_jsonl(draft_path)
    out = Path(out_path)
    done_ids: set[str] = set()
    if out.exists():
        done_ids = {r["id"] for r in read_jsonl(out)}
    skipped_path = out.with_suffix(".skipped.json")
    skipped: list[str] = json.loads(skipped_path.read_text()) if skipped_path.exists() else []
    todo = [d for d in drafts if d["id"] not in done_ids and d["id"] not in skipped]
    con.print(f"[bold]{len(done_ids)} verified, {len(skipped)} skipped, {len(todo)} to go[/bold]")
    stats = {"accepted_unchanged": 0, "accepted_edited": 0, "skipped": 0}
    for k, d in enumerate(todo):
        label = {**_blank(), **(d.get("label") or {})}
        edited: list[str] = []
        con.rule(f"{d['id']}  ({k + 1}/{len(todo)})")
        con.print(
            d["text"][:preview_chars]
            + (" [dim]...(v to view all)[/dim]" if len(d["text"]) > preview_chars else "")
        )
        while True:
            con.print(_table(label))
            cmd = input_fn("[a]ccept / [e]dit / [s]kip / [v]iew / [q]uit > ").strip().lower()
            if cmd == "v":
                con.print(d["text"])
            elif cmd == "q":
                return _finish(out, stats, verifier, con)
            elif cmd == "s":
                skipped.append(d["id"])
                skipped_path.write_text(json.dumps(skipped))
                stats["skipped"] += 1
                break
            elif cmd == "e":
                field = input_fn("field > ").strip()
                if field not in FIELD_ORDER:
                    con.print(f"[red]unknown field {field!r}[/red]")
                    continue
                raw = input_fn(f"{field} (JSON) > ").strip()
                try:
                    candidate = {**label, field: json.loads(raw)}
                    label = JobPosting.model_validate(candidate).to_json_dict()
                    edited.append(field)
                except (json.JSONDecodeError, ValidationError) as e:
                    con.print(f"[red]rejected: {e}[/red]")
            elif cmd == "a":
                label = JobPosting.model_validate(label).to_json_dict()
                row = {
                    "id": d["id"],
                    "text": d["text"],
                    "label": label,
                    "meta": {
                        **{k2: v for k2, v in d.items() if k2 not in ("id", "text", "label", "meta")},
                        **(d.get("meta") or {}),
                        "source": "real",
                        "verified": True,
                        "edited_fields": sorted(set(edited)),
                        "verified_by": verifier,
                        "verified_at": datetime.now(UTC).strftime("%Y-%m-%d"),
                    },
                }
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["accepted_edited" if edited else "accepted_unchanged"] += 1
                break
    return _finish(out, stats, verifier, con)


def _finish(out: Path, stats: dict[str, int], verifier: str, con: Console) -> dict[str, Any]:
    rows = read_jsonl(out) if out.exists() else []
    meta = {
        "n": len(rows),
        "dataset": rows[0]["meta"].get("dataset") if rows else None,
        "licence_declared": rows[0]["meta"].get("licence") if rows else None,
        "edited_any_field": sum(1 for r in rows if r["meta"].get("edited_fields")),
        "this_session": stats,
        "verified_by": verifier,
    }
    out.with_name("real_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    con.print(f"[green]saved {len(rows)} verified postings to {out}[/green]")
    return meta
