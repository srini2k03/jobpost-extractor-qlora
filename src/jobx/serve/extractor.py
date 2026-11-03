"""Extraction with schema validation and retry-on-invalid."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from jobx import prompts
from jobx.schema import json_schema, parse_output
from jobx.serve.llm import LLM

RETRY_MESSAGE = (
    "Your previous reply was not valid for the schema ({error}). Reply again with only the corrected JSON "
    "object, with all keys."
)


@dataclass
class ExtractResult:
    data: dict[str, Any] | None
    valid: bool
    attempts: int
    raw_outputs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    backend: str = ""


class Extractor:
    def __init__(
        self, llm: LLM, *, max_retries: int = 2, guided: bool = False, max_tokens: int = 384
    ) -> None:
        self.llm = llm
        self.max_retries = max_retries
        self.guided = guided
        self.max_tokens = max_tokens
        self._schema = json_schema()

    def extract(self, text: str, *, retry: bool = True, guided: bool | None = None) -> ExtractResult:
        t0 = time.perf_counter()
        msgs = prompts.messages(text)
        use_guided = self.guided if guided is None else guided
        res = ExtractResult(None, False, 0, backend=getattr(self.llm, "name", type(self.llm).__name__))
        attempts = 1 + (self.max_retries if retry else 0)
        for _ in range(attempts):
            out = self.llm.chat(msgs, schema=self._schema if use_guided else None, max_tokens=self.max_tokens)
            res.attempts += 1
            res.raw_outputs.append(out)
            pr = parse_output(out)
            if pr.schema_ok and pr.posting is not None:
                res.data = pr.posting.to_json_dict()
                res.valid = True
                break
            res.errors.append(pr.error or "invalid")
            if pr.json_ok and res.data is None:
                res.data = pr.data  # best effort, flagged valid=False
            msgs = [
                *msgs,
                {"role": "assistant", "content": out},
                {"role": "user", "content": RETRY_MESSAGE.format(error=pr.error)},
            ]
        res.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return res
