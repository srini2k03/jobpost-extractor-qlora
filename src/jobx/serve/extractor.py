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
    token_logprobs: list[dict[str, Any]] = field(
        default_factory=list
    )  # final attempt, when the backend has them
    raw_text: str = ""


class Extractor:
    def __init__(
        self,
        llm: LLM,
        *,
        max_retries: int = 2,
        guided: bool = False,
        max_tokens: int = 384,
        system: str = "compact",
    ) -> None:
        self.llm = llm
        self.max_retries = max_retries
        self.guided = guided
        self.max_tokens = max_tokens
        self.system = system  # "compact" for the CPU fine-tune (its training prompt), "full" for base models
        self._schema = json_schema()

    def extract(
        self, text: str, *, retry: bool = True, guided: bool | None = None, logprobs: bool = False
    ) -> ExtractResult:
        t0 = time.perf_counter()
        msgs = prompts.messages(text, system=self.system)
        use_guided = self.guided if guided is None else guided
        res = ExtractResult(None, False, 0, backend=getattr(self.llm, "name", type(self.llm).__name__))
        attempts = 1 + (self.max_retries if retry else 0)
        with_lp = logprobs and hasattr(self.llm, "chat_with_logprobs")
        for _ in range(attempts):
            schema = self._schema if use_guided else None
            if with_lp:
                out, toks = self.llm.chat_with_logprobs(msgs, schema=schema, max_tokens=self.max_tokens)  # type: ignore[attr-defined]
                res.token_logprobs = toks
            else:
                out = self.llm.chat(msgs, schema=schema, max_tokens=self.max_tokens)
            res.attempts += 1
            res.raw_outputs.append(out)
            res.raw_text = out
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
