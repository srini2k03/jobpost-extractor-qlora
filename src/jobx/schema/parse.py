"""Turn raw model text into (json_ok, schema_ok, lenient dict, validated posting)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from jobx.schema.models import FIELD_ORDER, JobPosting

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class ParseResult:
    json_ok: bool
    schema_ok: bool
    data: dict[str, Any]
    posting: JobPosting | None
    error: str | None
    truncated: bool = False


def _first_json_object(text: str) -> str | None:
    """Return the first balanced {...} block, honouring strings and escapes."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None  # unbalanced from this start: truncated output
    return None


def extract_json_text(text: str) -> str | None:
    m = _FENCE_RE.search(text)
    if m:
        inner = _first_json_object(m.group(1))
        if inner is not None:
            return inner
    return _first_json_object(text)


def parse_output(text: str) -> ParseResult:
    """Parse a model completion.

    ``json_ok`` - a JSON object could be parsed (code fences and chatter around it are allowed).
    ``schema_ok`` - the object validates against :class:`JobPosting` exactly (no extra keys).
    ``data`` - a lenient dict restricted to schema keys, used for field scoring even when the
    object is schema-invalid, so one bad field does not zero the others.
    """
    raw = extract_json_text(text or "")
    if raw is None:
        truncated = "{" in (text or "")
        return ParseResult(False, False, {}, None, "no JSON object found", truncated=truncated)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return ParseResult(False, False, {}, None, f"json decode error: {e}")
    if not isinstance(obj, dict):
        return ParseResult(False, False, {}, None, "top-level JSON is not an object")
    data = {k: obj.get(k) for k in FIELD_ORDER}
    try:
        posting = JobPosting.model_validate(obj)
    except ValidationError as e:
        return ParseResult(True, False, data, None, f"schema error: {e.errors()[0]['msg']}")
    return ParseResult(True, True, posting.to_json_dict(), posting, None)
