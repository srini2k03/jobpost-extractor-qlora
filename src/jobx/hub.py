"""Hugging Face Hub helpers: resolve the user's account and repo names, push/pull artefacts.

The account name is never hard-coded. Resolution order:
1. ``HF_USERNAME`` environment variable;
2. Colab secret ``HF_USERNAME`` (``google.colab.userdata``);
3. ``hf_username`` in ``configs/hub.yaml`` (placeholder ``YOUR_HF_USERNAME`` is ignored);
4. ``whoami()`` with the token (``HF_TOKEN`` env / Colab secret).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from jobx import REPO_ROOT

PLACEHOLDER = "YOUR_HF_USERNAME"


def _colab_secret(name: str) -> str | None:
    try:
        from google.colab import userdata

        v = userdata.get(name)
        return str(v) if v else None
    except Exception:  # not on Colab, or the secret is missing / not granted to this notebook
        return None


def secret(name: str) -> str | None:
    """Environment variable first, then Colab secret."""
    return os.environ.get(name) or _colab_secret(name)


def hub_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else REPO_ROOT / "configs" / "hub.yaml"
    if not p.exists():
        return {}
    cfg: dict[str, Any] = yaml.safe_load(p.read_text()) or {}
    return cfg


def hf_username(config_path: str | Path | None = None) -> str:
    name = secret("HF_USERNAME")
    if name:
        return name
    cfg_name = str(hub_config(config_path).get("hf_username") or "")
    if cfg_name and cfg_name != PLACEHOLDER:
        return cfg_name
    token = secret("HF_TOKEN")
    if token:
        from huggingface_hub import HfApi

        return str(HfApi(token=token).whoami()["name"])
    raise RuntimeError(
        "Hugging Face username unknown: set HF_USERNAME (env or Colab secret), edit configs/hub.yaml, "
        "or provide HF_TOKEN so it can be looked up."
    )


def repo_id(kind: str, config_path: str | Path | None = None) -> str:
    """kind is a key of ``repos`` in configs/hub.yaml, e.g. 'artifacts', 'adapter', 'merged', 'gguf'."""
    cfg = hub_config(config_path)
    name = cfg.get("repos", {}).get(kind)
    if not name:
        raise KeyError(f"no repo named {kind!r} in configs/hub.yaml")
    return f"{hf_username(config_path)}/{name}"


def push_folder(
    local_dir: str | Path,
    kind: str,
    path_in_repo: str = "",
    *,
    repo_type: str = "model",
    private: bool = False,
    commit_message: str = "upload",
) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=secret("HF_TOKEN"))
    rid = repo_id(kind)
    api.create_repo(rid, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=rid,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )
    return rid


def push_file(
    local_file: str | Path, kind: str, path_in_repo: str, *, repo_type: str = "dataset", private: bool = False
) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=secret("HF_TOKEN"))
    rid = repo_id(kind)
    api.create_repo(rid, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(local_file), path_in_repo=path_in_repo, repo_id=rid, repo_type=repo_type
    )
    return rid


def pull_artifacts(dest: str | Path = ".", patterns: list[str] | None = None) -> Path:
    """Download predictions/metrics/benchmarks/training logs from the artifacts dataset repo
    into the repository layout (results/..., data/...)."""
    from huggingface_hub import snapshot_download

    rid = repo_id("artifacts")
    path = snapshot_download(
        repo_id=rid,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=patterns or ["results/**", "data/paraphrased/**", "data/real/**"],
        token=secret("HF_TOKEN"),
    )
    return Path(path)
