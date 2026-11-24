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

    def chat_with_logprobs(
        self, messages: list[dict[str, str]], *, schema: dict[str, Any] | None = None, max_tokens: int = 384
    ) -> tuple[str, list[dict[str, Any]]]:
        """Like :meth:`chat`, plus llama.cpp's per-token log-probs (used for field confidence)."""
        kwargs: dict[str, Any] = {
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "logprobs": True,
            "top_logprobs": 1,
        }
        if schema is not None:
            kwargs["response_format"] = llama_cpp_response_format(schema)
        out = self.llm.create_chat_completion(messages=messages, **kwargs)
        ch = out["choices"][0]
        toks = list(((ch.get("logprobs") or {}).get("content")) or [])
        return str(ch["message"]["content"] or ""), toks


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

    def __init__(
        self,
        responses: Sequence[str] | Callable[[list[dict[str, str]]], str],
        *,
        logprob: float | None = None,
    ) -> None:
        self._fn = responses if callable(responses) else None
        self._queue = list(responses) if not callable(responses) else []
        self.calls: list[list[dict[str, str]]] = []
        self.logprob = logprob  # when set, chat_with_logprobs returns 4-character tokens with this log-prob

    def chat(
        self, messages: list[dict[str, str]], *, schema: dict[str, Any] | None = None, max_tokens: int = 384
    ) -> str:
        self.calls.append(messages)
        if self._fn is not None:
            return self._fn(messages)
        return self._queue.pop(0) if self._queue else ""

    def chat_with_logprobs(
        self, messages: list[dict[str, str]], *, schema: dict[str, Any] | None = None, max_tokens: int = 384
    ) -> tuple[str, list[dict[str, Any]]]:
        text = self.chat(messages, schema=schema, max_tokens=max_tokens)
        if self.logprob is None:
            return text, []
        toks = [text[i : i + 4] for i in range(0, len(text), 4)]
        return text, [{"token": t, "bytes": list(t.encode()), "logprob": self.logprob} for t in toks]


def prompt_from_env() -> str:
    """JOBX_PROMPT=compact (default: the CPU fine-tune's training prompt) or full (base models, GPU route)."""
    name = os.environ.get("JOBX_PROMPT", "compact").lower()
    if name not in ("compact", "full"):
        raise ValueError(f"JOBX_PROMPT must be compact or full, got {name!r}")
    return name


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
