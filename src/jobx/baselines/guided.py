"""Guided (constrained) JSON decoding.

* transformers: lm-format-enforcer ``prefix_allowed_tokens_fn`` (used in notebooks 02/04);
* llama.cpp: ``response_format={"type": "json_object", "schema": ...}`` (server + Space).

Both get a ref-free copy of the schema because some constraint engines do not resolve ``$ref``.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``schema`` with local ``#/$defs/...`` references inlined."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and isinstance(node["$ref"], str) and node["$ref"].startswith("#/$defs/"):
                target = copy.deepcopy(defs[node["$ref"].split("/")[-1]])
                extra = {k: v for k, v in node.items() if k != "$ref"}
                return walk({**target, **extra})
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    out: dict[str, Any] = walk(schema)
    return out


def require_all(schema: dict[str, Any]) -> dict[str, Any]:
    """Mark every property of every object as required (and forbid extras).

    The Pydantic schema gives fields defaults, so they are optional; a grammar built from it would accept
    ``{}``. Requiring every key makes constrained decoding emit the full record, with ``null`` / ``[]``
    for unstated fields, exactly the training target format.
    """

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out = {k: walk(v) for k, v in node.items()}
            if isinstance(out.get("properties"), dict):
                out["required"] = list(out["properties"])
                out["additionalProperties"] = False
            return out
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    result: dict[str, Any] = walk(schema)
    return result


# Decoding-only caps on list lengths. Without them a greedy 0.5B model can loop on one list item ("sap fico",
# "sap fico", ...) until the token limit and the JSON is never closed (5 of 130 real postings in the first
# test_real run). The largest gold lists are 19 skills and 5 locations, so the caps never cut a correct answer.
LIST_CAPS: dict[str, int] = {"skills": 30, "locations": 10}


def cap_lists(schema: dict[str, Any], caps: dict[str, int] | None = None) -> dict[str, Any]:
    out = copy.deepcopy(schema)
    for name, n in (caps or LIST_CAPS).items():
        prop = out.get("properties", {}).get(name)
        if isinstance(prop, dict) and prop.get("type") == "array":
            prop["maxItems"] = n
    return out


def guided_schema() -> dict[str, Any]:
    """The schema every constrained decoder uses: JobPosting, refs inlined, all keys required, list caps."""
    from jobx.schema import json_schema

    return cap_lists(require_all(inline_refs(json_schema())))


def build_transformers_prefix_fn(tokenizer: Any, schema: dict[str, Any]) -> Callable[..., Any]:
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

    parser = JsonSchemaParser(require_all(inline_refs(schema)))
    return build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)


def llama_cpp_response_format(schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "json_object", "schema": cap_lists(require_all(inline_refs(schema)))}
