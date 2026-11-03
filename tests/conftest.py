from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _repo_root_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Commands and default paths are relative to the repository root."""
    monkeypatch.chdir(ROOT)
    for k in ("HF_TOKEN", "HF_USERNAME", "JOBX_BACKEND", "JOBX_GGUF_PATH", "JOBX_GGUF_REPO"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="session")
def small_generated() -> list[dict]:
    import yaml

    from jobx.data.generate import GenConfig, generate

    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    cfg["n_examples"] = 400
    cfg["seed"] = 99
    return generate(GenConfig.from_dict(cfg))


def gold_example() -> dict:
    return {
        "company": "Kaveri Analytics",
        "role": "Data Engineer",
        "seniority": "senior",
        "skills": ["python", "spark", "aws"],
        "min_experience_years": 5,
        "max_experience_years": 8,
        "salary": {"min": 1800000, "max": 2400000, "currency": "INR", "period": "year"},
        "locations": ["Bengaluru", "Pune"],
        "work_mode": "hybrid",
        "employment_type": "full_time",
        "education": "b.tech",
    }


os.environ.setdefault("PYTHONHASHSEED", "0")
