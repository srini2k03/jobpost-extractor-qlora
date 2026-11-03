"""LLM backends for serving. All return the raw assistant text for a list of chat messages."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from jobx.baselines.guided import llama_cpp_response_format


class LLM(Protocol):
    name: str

    def chat(
        self, messages: list[dict[str, str]], *, schema: dict[str, Any] | None = None, max_tokens: int = 384
    ) -> str: ...


class LlamaCppLLM:
    """GGUF model on CPU through llama-cpp-python."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        repo_id: str | None = None,
        filename: str | None = None,
        n_ctx: int = 4096,
        n_threads: int | None = None,
    ) -> None:
        from llama_cpp import Llama

        common: dict[str, Any] = {"n_ctx": n_ctx, "n_threads": n_threads or os.cpu_count(), "verbose": False}
        if model_path:
            self.llm = Llama(model_path=model_path, **common)
            self.name = os.path.basename(model_path)
        elif repo_id and filename:
            self.llm = Llama.from_pretrained(repo_id=repo_id, filename=filename, **common)
            self.name = f"{repo_id}/{filename}"
        else:
            raise ValueError("need model_path or repo_id + filename")

    def chat(
        self, messages: list[dict[str, str]], *, schema: dict[str, Any] | None = None, max_tokens: int = 384
    ) -> str:
        kwargs: dict[str, Any] = {"temperature": 0.0, "max_tokens": max_tokens}
        if schema is not None:
            kwargs["response_format"] = llama_cpp_response_format(schema)
        out = self.llm.create_chat_completion(messages=messages, **kwargs)
        return str(out["choices"][0]["message"]["content"] or "")


class RegexLLM:
    """Model-free fallback that answers with the regex baseline (demo / smoke tests)."""

    name = "regex-baseline"

    def chat(
        self, messages: list[dict[str, str]], *, schema: dict[str, Any] | None = None, max_tokens: int = 384
    ) -> str:
        from jobx.baselines.regex_extractor import predict

        user = messages[-1]["content"]
        text = user.split("<posting>", 1)[-1].split("</posting>", 1)[0]
        return predict(text)


class MockLLM:
    """Scripted responses for tests: a list (consumed in order) or a function of the messages."""

    name = "mock"

    def __init__(self, responses: Sequence[str] | Callable[[list[dict[str, str]]], str]) -> None:
        self._fn = responses if callable(responses) else None
        self._queue = list(responses) if not callable(responses) else []
        self.calls: list[list[dict[str, str]]] = []

    def chat(
        self, messages: list[dict[str, str]], *, schema: dict[str, Any] | None = None, max_tokens: int = 384
    ) -> str:
        self.calls.append(messages)
        if self._fn is not None:
            return self._fn(messages)
        return self._queue.pop(0) if self._queue else ""


def llm_from_env() -> LLM:
    """JOBX_BACKEND=regex | gguf (default gguf). For gguf: JOBX_GGUF_PATH, or JOBX_GGUF_REPO + JOBX_GGUF_FILE."""
    backend = os.environ.get("JOBX_BACKEND", "gguf").lower()
    if backend == "regex":
        return RegexLLM()
    path = os.environ.get("JOBX_GGUF_PATH")
    if path:
        return LlamaCppLLM(path)
    repo = os.environ.get("JOBX_GGUF_REPO")
    if not repo:
        from jobx.hub import repo_id

        repo = repo_id("gguf")
    return LlamaCppLLM(repo_id=repo, filename=os.environ.get("JOBX_GGUF_FILE", "*Q4_K_M.gguf"))
