"""FastAPI service: messy job posting in, schema-valid JSON with evidence and confidence out.

Run locally:  JOBX_GGUF_PATH=jobx-qwen2.5-0.5b-Q4_K_M.gguf uvicorn jobx.serve.app:app --port 8000
Model-free:   JOBX_BACKEND=regex uvicorn jobx.serve.app:app

Endpoints (OpenAPI docs at /docs):
  GET  /health          liveness + whether the model is loaded
  GET  /model           backend, prompt, limits, calibration and evaluation summary
  GET  /schema          JSON Schema of the output
  POST /extract         one posting -> JSON + evidence spans + (calibrated) field confidence + abstention
  POST /extract/batch   up to JOBX_MAX_BATCH postings
  POST /validate        check a JSON object against the schema (optionally with evidence for a text)
  POST /evidence        evidence spans of a given record in a given text (no model call)

Inputs are untrusted: postings are passed to the model inside <posting> tags as data, size-limited, and
never written to the logs (only lengths, timings and outcomes are logged).
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field, ValidationError

from jobx import REPO_ROOT, __version__
from jobx.evidence import evidence_by_field, find_evidence, missing_cues, summarise
from jobx.schema import FIELD_ORDER, JobPosting, json_schema
from jobx.serve.config import Settings
from jobx.serve.extractor import Extractor, ExtractResult
from jobx.serve.llm import LLM, llm_from_env, prompt_from_env
from jobx.uncertainty import Calibrator, apply_abstention, field_features, field_logprobs

log = logging.getLogger("jobx.serve")
MAX_CHARS = Settings().max_chars  # default per-posting limit (JOBX_MAX_CHARS overrides it at runtime)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {"level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            base.update(extra)
        return json.dumps(base, ensure_ascii=False)


def _setup_logging(level: str) -> None:
    if not any(isinstance(h.formatter, JsonFormatter) for h in log.handlers):
        h = logging.StreamHandler()
        h.setFormatter(JsonFormatter())
        log.addHandler(h)
    log.setLevel(level)
    log.propagate = False


class RateLimiter:
    """Sliding one-minute window per client key (in memory; one process)."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        if self.per_minute <= 0:
            return True
        t = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and t - q[0] >= 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(t)
            return True


Abstain = Literal["off", "flag", "null"]


class ExtractOptions(BaseModel):
    retry: bool = Field(True, description="Re-prompt the model when its output is not schema-valid")
    guided: bool | None = Field(None, description="Constrain decoding to the JSON schema (llama.cpp grammar)")
    evidence: bool = Field(True, description="Return the supporting text span of every extracted value")
    abstain: Abstain = Field(
        "off",
        description="off: keep all values; flag: list low-confidence fields; null: also replace them with "
        "null/[] (the withheld values are returned separately)",
    )
    min_confidence: float | None = Field(
        None,
        ge=0,
        le=1,
        description="Abstention threshold; default: the threshold calibrated on the validation split",
    )


class ExtractRequest(ExtractOptions):
    text: str = Field(..., min_length=1, description="Raw job posting text (untrusted input)")


class BatchItem(BaseModel):
    id: str | None = Field(None, max_length=128)
    text: str = Field(..., min_length=1)


class BatchRequest(ExtractOptions):
    items: list[BatchItem] = Field(..., min_length=1)


class SpanOut(BaseModel):
    start: int
    end: int
    text: str


class EvidenceOut(BaseModel):
    field: str
    value: Any
    status: Literal["supported", "weak", "unsupported"]
    match: str
    spans: list[SpanOut]


class ExtractResponse(BaseModel):
    data: dict[str, Any] | None = Field(
        None, description="Schema-valid JobPosting (canonical key order), or null"
    )
    valid: bool
    attempts: int
    errors: list[str]
    latency_ms: float
    backend: str
    evidence: list[EvidenceOut] = Field(default_factory=list)
    evidence_summary: dict[str, Any] | None = None
    possible_omissions: dict[str, SpanOut] = Field(
        default_factory=dict, description="Null fields whose value the text may state (a hint, not proof)"
    )
    confidence: dict[str, float] = Field(
        default_factory=dict,
        description="Calibrated P(field correct); empty when the backend has no log-probs",
    )
    confidence_source: Literal["calibrated", "evidence", "none"] = "none"
    threshold: float | None = None
    uncertain_fields: list[str] = Field(default_factory=list)
    withheld: dict[str, Any] = Field(
        default_factory=dict, description="Values replaced by null (abstain='null')"
    )


class BatchResponse(BaseModel):
    results: list[dict[str, Any]]
    latency_ms: float


class ValidateRequest(BaseModel):
    data: dict[str, Any]
    text: str | None = Field(
        None, description="If given, evidence spans of the record in this text are returned"
    )


class EvidenceRequest(BaseModel):
    text: str = Field(..., min_length=1)
    data: dict[str, Any]


@lru_cache(maxsize=4)
def _load_calibrator(path: str) -> tuple[Calibrator | None, float | None, str | None]:
    p = Path(path)
    if not p.exists():
        return None, None, None
    d = json.loads(p.read_text())
    return Calibrator.from_dict(d), float(d.get("threshold", 0.5)), d.get("approach")


def _eval_summary() -> dict[str, Any]:
    rj = REPO_ROOT / "results/results.json"
    if not rj.exists():
        return {}
    data = json.loads(rj.read_text())
    out: dict[str, Any] = {}
    for split, rows in data.get("splits", {}).items():
        for r in rows:
            if r.get("approach") == "qwen2.5-0.5b-lora-cpu_q4_k_m" and r.get("status") == "measured":
                out[split] = {"n": r.get("n"), "micro_f1": r["metrics"].get("micro_f1")}
    return out


def create_app(
    llm: LLM | None = None, *, max_retries: int | None = None, settings: Settings | None = None
) -> FastAPI:
    cfg = settings or Settings.from_env()
    retries = cfg.max_retries if max_retries is None else max_retries
    _setup_logging(cfg.log_level)
    app = FastAPI(
        title="jobx extractor",
        version=__version__,
        description="Messy Indian job postings to strict JSON, with evidence spans and field confidence.",
    )
    limiter = RateLimiter(cfg.rate_limit_per_min)

    def _make(backend: LLM) -> Extractor:
        return Extractor(backend, max_retries=retries, max_tokens=cfg.max_tokens, system=prompt_from_env())

    state: dict[str, Any] = {"extractor": _make(llm) if llm else None}

    def extractor() -> Extractor:
        if state["extractor"] is None:
            state["extractor"] = _make(llm_from_env())
        ex: Extractor = state["extractor"]
        return ex

    @app.middleware("http")
    async def access_log(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        client = request.client.host if request.client else "unknown"
        if request.method == "POST" and not limiter.allow(client):
            log.warning("rate limited", extra={"fields": {"request_id": rid, "path": request.url.path}})
            return Response(
                json.dumps({"detail": f"rate limit: {cfg.rate_limit_per_min} requests per minute"}),
                status_code=429,
                media_type="application/json",
                headers={"retry-after": "60", "x-request-id": rid},
            )
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        log.info(
            "request",
            extra={
                "fields": {
                    "request_id": rid,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "ms": round((time.perf_counter() - t0) * 1000, 1),
                    "bytes_in": int(request.headers.get("content-length") or 0),
                }
            },
        )
        return response

    def _check_len(text: str) -> None:
        if len(text) > cfg.max_chars:
            raise HTTPException(status_code=413, detail=f"posting longer than {cfg.max_chars} characters")

    def _run(text: str, opts: ExtractOptions) -> ExtractResponse:
        ex = extractor()
        r: ExtractResult = ex.extract(text, retry=opts.retry, guided=opts.guided, logprobs=True)
        data = JobPosting.model_validate(r.data).to_json_dict() if r.valid and r.data is not None else None
        out = ExtractResponse(
            data=data,
            valid=r.valid,
            attempts=r.attempts,
            errors=r.errors,
            latency_ms=r.latency_ms,
            backend=r.backend,
        )
        if data is None:
            return out
        evs = find_evidence(text, data)
        if opts.evidence:
            out.evidence = [EvidenceOut.model_validate(e.as_dict()) for e in evs]
            out.evidence_summary = summarise(evs)
            out.possible_omissions = {
                k: SpanOut.model_validate(v.__dict__) for k, v in missing_cues(text, data).items()
            }
        ev_status = evidence_by_field(evs)
        cal, cal_thr, _ = _load_calibrator(str(cfg.calibrator_path))
        threshold = opts.min_confidence if opts.min_confidence is not None else cfg.default_threshold
        lps = field_logprobs(r.token_logprobs, r.raw_text) if r.token_logprobs else {}
        conf: dict[str, float] = {}
        if cal is not None and lps:
            feats = np.array(
                [field_features(lps.get(f), ev_status.get(f), data[f] in (None, [])) for f in FIELD_ORDER]
            )
            probs = cal.predict(feats, list(FIELD_ORDER))
            conf = {
                f: round(float(p), 4)
                for f, p in zip(FIELD_ORDER, probs, strict=True)
                if not math.isnan(float(p))
            }
            out.confidence_source = "calibrated"
            threshold = (cal_thr if cal_thr is not None else 0.5) if threshold is None else threshold
        else:  # no log-probs or no calibrator: evidence support only (1 supported, 0.5 weak, 0 unsupported)
            score = {"supported": 1.0, "weak": 0.5, "unsupported": 0.0}
            conf = {f: score[ev_status[f]] for f in ev_status}
            out.confidence_source = "evidence"
            threshold = 0.5 if threshold is None else threshold
        out.confidence = conf
        out.threshold = threshold
        if opts.abstain != "off":
            new, uncertain, withheld = apply_abstention(data, conf, threshold=threshold, mode=opts.abstain)
            out.data, out.uncertain_fields, out.withheld = new, uncertain, withheld
        log.info(
            "extract",
            extra={
                "fields": {
                    "chars": len(text),
                    "valid": r.valid,
                    "attempts": r.attempts,
                    "latency_ms": r.latency_ms,
                    "unsupported": sum(1 for e in evs if e.status == "unsupported"),
                    "uncertain": len(out.uncertain_fields),
                }
            },
        )
        return out

    @app.get("/health")
    def health() -> dict[str, Any]:
        loaded = state["extractor"] is not None
        return {
            "status": "ok",
            "model_loaded": loaded,
            "backend": state["extractor"].llm.name if loaded else None,
        }

    @app.get("/model")
    def model() -> dict[str, Any]:
        cal, thr, approach = _load_calibrator(str(cfg.calibrator_path))
        ex = state["extractor"]
        return {
            "version": __version__,
            "backend": ex.llm.name if ex else None,
            "prompt": ex.system if ex else prompt_from_env(),
            "schema_fields": list(FIELD_ORDER),
            "limits": {
                "max_chars": cfg.max_chars,
                "max_batch": cfg.max_batch,
                "max_batch_chars": cfg.max_batch_chars,
                "rate_limit_per_min": cfg.rate_limit_per_min,
            },
            "calibration": {"fitted_for": approach, "threshold": thr, "n_train": cal.n_train}
            if cal
            else None,
            "evaluation_q4_k_m": _eval_summary(),
        }

    @app.get("/schema")
    def schema() -> dict[str, Any]:
        return json_schema()

    @app.post("/extract", response_model=ExtractResponse)
    def extract(req: ExtractRequest) -> ExtractResponse:
        _check_len(req.text)
        return _run(req.text, req)

    @app.post("/extract/batch", response_model=BatchResponse)
    def extract_batch(req: BatchRequest) -> BatchResponse:
        if len(req.items) > cfg.max_batch:
            raise HTTPException(status_code=413, detail=f"at most {cfg.max_batch} postings per batch")
        if sum(len(i.text) for i in req.items) > cfg.max_batch_chars:
            raise HTTPException(status_code=413, detail=f"batch longer than {cfg.max_batch_chars} characters")
        t0 = time.perf_counter()
        results = []
        for k, item in enumerate(req.items):
            rid = item.id or str(k)
            if len(item.text) > cfg.max_chars:
                results.append({"id": rid, "error": f"posting longer than {cfg.max_chars} characters"})
                continue
            results.append({"id": rid, **_run(item.text, req).model_dump()})
        return BatchResponse(results=results, latency_ms=round((time.perf_counter() - t0) * 1000, 1))

    @app.post("/validate")
    def validate(req: ValidateRequest) -> dict[str, Any]:
        try:
            posting = JobPosting.model_validate(req.data)
        except ValidationError as e:
            errs = [{"loc": ".".join(str(x) for x in err["loc"]), "msg": err["msg"]} for err in e.errors()]
            return {"valid": False, "errors": errs, "data": None}
        data = posting.to_json_dict()
        out: dict[str, Any] = {"valid": True, "errors": [], "data": data}
        if req.text:
            _check_len(req.text)
            evs = find_evidence(req.text, data)
            out["evidence"] = [e.as_dict() for e in evs]
            out["evidence_summary"] = summarise(evs)
        return out

    @app.post("/evidence")
    def evidence(req: EvidenceRequest) -> dict[str, Any]:
        _check_len(req.text)
        evs = find_evidence(req.text, req.data)
        return {"evidence": [e.as_dict() for e in evs], "summary": summarise(evs)}

    return app


app = create_app()
