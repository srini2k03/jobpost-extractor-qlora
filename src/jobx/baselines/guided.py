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


def build_transformers_prefix_fn(tokenizer: Any, schema: dict[str, Any]) -> Callable[..., Any]:
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

    parser = JsonSchemaParser(inline_refs(schema))
    return build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)


def llama_cpp_response_format(schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "json_object", "schema": inline_refs(schema)}
