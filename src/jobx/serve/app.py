"""FastAPI service: POST /extract turns a posting into validated JSON.

Run locally:  JOBX_GGUF_PATH=model.gguf uvicorn jobx.serve.app:app --port 8000
Model-free:   JOBX_BACKEND=regex uvicorn jobx.serve.app:app
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from jobx import __version__
from jobx.schema import JobPosting, json_schema
from jobx.serve.extractor import Extractor
from jobx.serve.llm import LLM, llm_from_env

MAX_CHARS = 8000


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw job posting text")
    retry: bool = Field(True, description="Re-prompt the model when its output is not schema-valid")
    guided: bool | None = Field(None, description="Constrain decoding to the JSON schema (llama.cpp grammar)")


class ExtractResponse(BaseModel):
    data: dict[str, Any] | None = Field(
        None, description="Schema-valid JobPosting (canonical key order), or null"
    )
    valid: bool
    attempts: int
    errors: list[str]
    latency_ms: float
    backend: str


def create_app(llm: LLM | None = None, *, max_retries: int = 2) -> FastAPI:
    app = FastAPI(
        title="jobx extractor", version=__version__, description="Messy Indian job postings to strict JSON."
    )
    state: dict[str, Any] = {"extractor": Extractor(llm, max_retries=max_retries) if llm else None}

    def extractor() -> Extractor:
        if state["extractor"] is None:
            state["extractor"] = Extractor(llm_from_env(), max_retries=max_retries)
        ex: Extractor = state["extractor"]
        return ex

    @app.get("/health")
    def health() -> dict[str, Any]:
        loaded = state["extractor"] is not None
        return {
            "status": "ok",
            "model_loaded": loaded,
            "backend": state["extractor"].llm.name if loaded else None,
        }

    @app.get("/schema")
    def schema() -> dict[str, Any]:
        return json_schema()

    @app.post("/extract", response_model=ExtractResponse)
    def extract(req: ExtractRequest) -> ExtractResponse:
        if len(req.text) > MAX_CHARS:
            raise HTTPException(status_code=413, detail=f"posting longer than {MAX_CHARS} characters")
        r = extractor().extract(req.text, retry=req.retry, guided=req.guided)
        data = JobPosting.model_validate(r.data).to_json_dict() if r.valid and r.data is not None else None
        return ExtractResponse(
            data=data,
            valid=r.valid,
            attempts=r.attempts,
            errors=r.errors,
            latency_ms=r.latency_ms,
            backend=r.backend,
        )

    return app


app = create_app()
