"""Field-level confidence, calibration and abstention.

Signals per field of one extraction:

* **token log-probs** of the generated value (llama.cpp ``logprobs``): mean and minimum over the tokens
  that spell the field's value in the JSON output (:func:`field_logprobs`);
* **evidence support** from :mod:`jobx.evidence` (supported / weak / unsupported; nulls have none);
* whether the value is null.

Raw token probabilities are not treated as calibrated. :class:`Calibrator` fits a small logistic model
(numpy, L2-regularised, Newton steps) on the *validation* split, mapping the signals to P(field correct).
:func:`reliability` measures calibration (ECE, Brier, AUROC, reliability bins) on held-out data, and
:func:`risk_coverage` gives selective accuracy when low-confidence fields are abstained on.

Abstention (:func:`apply_abstention`) is configurable: ``off`` keeps every value; ``flag`` keeps values
and marks fields below the threshold as uncertain; ``null`` replaces them with null / [] but keeps the
value and its evidence text in the response, so nothing is silently lost.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from jobx.schema.models import FIELD_ORDER, LIST_FIELDS
from jobx.schema.parse import extract_json_text

AbstainMode = Literal["off", "flag", "null"]
EVIDENCE_LEVELS = ("supported", "weak", "unsupported", "null")
FEATURES = ("mean_lp", "min_lp", "has_lp", "ev_supported", "ev_weak", "ev_unsupported", "is_null")


# ----------------------------------------------------------------------------------------------------------
# JSON value spans and token log-probs


def value_spans(text: str) -> dict[str, tuple[int, int]]:
    """Character spans of the top-level values (and list elements as ``key[i]``) of the JSON object in
    ``text``. Spans index into ``text`` itself."""
    raw = extract_json_text(text or "")
    if raw is None:
        return {}
    base = text.find(raw)
    dec = json.JSONDecoder()
    spans: dict[str, tuple[int, int]] = {}
    i = 1
    n = len(raw)

    def skip(j: int) -> int:
        while j < n and raw[j] in " \t\r\n,":
            j += 1
        return j

    while True:
        i = skip(i)
        if i >= n or raw[i] == "}":
            break
        try:
            key, i = dec.raw_decode(raw, i)
        except json.JSONDecodeError:
            break
        i = skip(i)
        if i >= n or raw[i] != ":":
            break
        i = skip(i + 1)
        try:
            val, end = dec.raw_decode(raw, i)
        except json.JSONDecodeError:
            break
        spans[str(key)] = (base + i, base + end)
        if isinstance(val, list):
            j = i + 1
            for k in range(len(val)):
                j = skip(j)
                try:
                    _, e = dec.raw_decode(raw, j)
                except json.JSONDecodeError:
                    break
                spans[f"{key}[{k}]"] = (base + j, base + e)
                j = e
        i = end
    return spans


def _token_offsets(tokens: Sequence[Mapping[str, Any]]) -> tuple[str, list[tuple[int, int, float]]]:
    """Text and (char_start, char_end, logprob) per token, using byte values when present so multi-byte
    characters split across tokens are placed correctly."""
    out: list[tuple[int, int, float]] = []
    buf = b""
    for t in tokens:
        b = bytes(t["bytes"]) if t.get("bytes") is not None else str(t.get("token", "")).encode()
        start = len(buf.decode("utf-8", errors="ignore"))
        buf += b
        end = len(buf.decode("utf-8", errors="ignore"))
        out.append((start, end, float(t.get("logprob", 0.0))))
    return buf.decode("utf-8", errors="replace"), out


def field_logprobs(
    tokens: Sequence[Mapping[str, Any]], text: str | None = None
) -> dict[str, dict[str, float]]:
    """Mean / min / sum log-prob and token count of the tokens spelling each field's value (and each
    list element). ``tokens`` is llama.cpp's ``choices[0].logprobs.content``."""
    gen, offs = _token_offsets(tokens)
    text = gen if text is None else text
    if text != gen:  # content and token stream disagree (e.g. stripped whitespace): align on a prefix
        shift = gen.find(text.strip()[:20]) if text.strip() else -1
        if shift < 0:
            return {}
        offs = [(s - shift, e - shift, lp) for s, e, lp in offs]
    out: dict[str, dict[str, float]] = {}
    for key, (s, e) in value_spans(text).items():
        lps = [lp for ts, te, lp in offs if ts < e and s < te]
        if lps:
            out[key] = {
                "mean": round(float(np.mean(lps)), 4),
                "min": round(float(np.min(lps)), 4),
                "sum": round(float(np.sum(lps)), 4),
                "n": len(lps),
            }
    return out


# ----------------------------------------------------------------------------------------------------------
# Features, calibration, abstention


def field_features(lp: Mapping[str, float] | None, evidence_status: str | None, is_null: bool) -> list[float]:
    has = lp is not None
    mean = float(lp["mean"]) if has and lp else 0.0
    mn = float(lp["min"]) if has and lp else 0.0
    ev = "null" if is_null else (evidence_status or "unsupported")
    return [
        mean,
        mn,
        1.0 if has else 0.0,
        1.0 if ev == "supported" else 0.0,
        1.0 if ev == "weak" else 0.0,
        1.0 if ev == "unsupported" else 0.0,
        1.0 if is_null else 0.0,
    ]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


@dataclass
class Calibrator:
    """Logistic model P(correct | features); one weight vector shared by all fields plus a per-field
    intercept, fitted on the validation split."""

    weights: list[float] = field(default_factory=list)
    field_bias: dict[str, float] = field(default_factory=dict)
    l2: float = 1.0
    n_train: int = 0
    features: tuple[str, ...] = FEATURES

    def _design(self, X: np.ndarray, fields: Sequence[str]) -> np.ndarray:
        onehot = np.array([[1.0 if f == g else 0.0 for g in FIELD_ORDER] for f in fields])
        return np.hstack([X, onehot])

    def fit(self, X: np.ndarray, y: np.ndarray, fields: Sequence[str], *, iters: int = 50) -> Calibrator:
        A = self._design(np.asarray(X, dtype=float), fields)
        w = np.zeros(A.shape[1])
        reg = np.full(A.shape[1], self.l2)
        reg[-len(FIELD_ORDER) :] = self.l2 * 0.1  # intercepts: light shrinkage
        for _ in range(iters):
            p = _sigmoid(A @ w)
            g = A.T @ (p - y) + reg * w
            H = (A * (p * (1 - p))[:, None]).T @ A + np.diag(reg)
            step = np.linalg.solve(H, g)
            w -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        k = len(self.features)
        self.weights = [round(float(v), 6) for v in w[:k]]
        self.field_bias = {f: round(float(v), 6) for f, v in zip(FIELD_ORDER, w[k:], strict=True)}
        self.n_train = len(y)
        return self

    def predict(self, X: np.ndarray, fields: Sequence[str]) -> np.ndarray:
        w = np.array(self.weights + [self.field_bias.get(f, 0.0) for f in FIELD_ORDER])
        return _sigmoid(self._design(np.asarray(X, dtype=float), fields) @ w)

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "weights": self.weights,
            "field_bias": self.field_bias,
            "l2": self.l2,
            "n_train": self.n_train,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Calibrator:
        return cls(
            weights=list(d["weights"]),
            field_bias=dict(d["field_bias"]),
            l2=float(d.get("l2", 1.0)),
            n_train=int(d.get("n_train", 0)),
            features=tuple(d.get("features", FEATURES)),
        )


def reliability(p: np.ndarray, y: np.ndarray, *, bins: int = 10) -> dict[str, Any]:
    """Expected calibration error, Brier score, AUROC (correct vs wrong) and the reliability table."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(p) == 0:
        return {"n": 0}
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    table = []
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(conf - acc)
        table.append(
            {
                "bin": [round(edges[b], 2), round(edges[b + 1], 2)],
                "n": int(m.sum()),
                "conf": round(conf, 4),
                "acc": round(acc, 4),
            }
        )
    return {
        "n": len(p),
        "accuracy": round(float(y.mean()), 4),
        "mean_confidence": round(float(p.mean()), 4),
        "ece": round(float(ece), 4),
        "brier": round(float(np.mean((p - y) ** 2)), 4),
        "auroc": auroc(p, y),
        "bins": table,
    }


def auroc(score: np.ndarray, y: np.ndarray) -> float | None:
    """Probability that a random correct item scores higher than a random wrong one (ties count half)."""
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):  # average ranks for ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    rp = r[: len(pos)].sum()
    return round(float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 4)


def risk_coverage(p: np.ndarray, y: np.ndarray, thresholds: Sequence[float]) -> list[dict[str, float]]:
    """Coverage (share of fields kept) and selective accuracy when fields with p < t are abstained on."""
    out = []
    for t in thresholds:
        keep = p >= t
        out.append(
            {
                "threshold": round(float(t), 4),
                "coverage": round(float(keep.mean()), 4) if len(p) else 0.0,
                "selective_accuracy": round(float(y[keep].mean()), 4) if keep.any() else math.nan,
            }
        )
    return out


def choose_threshold(p: np.ndarray, y: np.ndarray, target_accuracy: float) -> float:
    """Smallest threshold whose selective accuracy on (validation) data reaches ``target_accuracy``."""
    for t in np.unique(np.round(p, 3)):
        keep = p >= t
        if keep.any() and y[keep].mean() >= target_accuracy:
            return float(t)
    return 1.0


def apply_abstention(
    data: Mapping[str, Any], confidence: Mapping[str, float], *, threshold: float, mode: AbstainMode = "flag"
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Returns (data after abstention, uncertain fields, values withheld by mode='null')."""
    uncertain = [f for f in FIELD_ORDER if f in confidence and confidence[f] < threshold]
    out = dict(data)
    withheld: dict[str, Any] = {}
    if mode == "null":
        for f in uncertain:
            withheld[f] = out.get(f)
            out[f] = [] if f in LIST_FIELDS else None
    if mode == "off":
        uncertain = []
    return out, uncertain, withheld
