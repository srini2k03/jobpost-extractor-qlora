"""Service settings, read once from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from jobx import REPO_ROOT


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def _float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    max_chars: int = 8000  # per posting
    max_batch: int = 16  # postings per /extract/batch request
    max_batch_chars: int = 64000  # total characters per batch request
    rate_limit_per_min: int = 60  # requests per client IP per minute; 0 disables
    max_retries: int = 2  # re-prompts when the output is not schema-valid
    max_tokens: int = 384
    calibrator_path: Path = REPO_ROOT / "results/calibration/calibrator.json"
    default_threshold: float | None = None  # overrides the calibrated abstention threshold
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()
        return cls(
            max_chars=_int("JOBX_MAX_CHARS", d.max_chars),
            max_batch=_int("JOBX_MAX_BATCH", d.max_batch),
            max_batch_chars=_int("JOBX_MAX_BATCH_CHARS", d.max_batch_chars),
            rate_limit_per_min=_int("JOBX_RATE_LIMIT_PER_MIN", d.rate_limit_per_min),
            max_retries=_int("JOBX_MAX_RETRIES", d.max_retries),
            max_tokens=_int("JOBX_MAX_TOKENS", d.max_tokens),
            calibrator_path=Path(os.environ.get("JOBX_CALIBRATOR", str(d.calibrator_path))),
            default_threshold=_float("JOBX_ABSTAIN_THRESHOLD", d.default_threshold),
            log_level=os.environ.get("JOBX_LOG_LEVEL", d.log_level).upper(),
        )
