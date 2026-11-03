"""jobx: messy Indian job postings -> strict JSON."""

from pathlib import Path

__version__ = "0.1.0"

_HERE = Path(__file__).resolve()
# Repository root in a source checkout (src/jobx -> repo). In the Space bundle jobx/ sits next to
# configs/, so the parent directory is the root there.
REPO_ROOT = _HERE.parents[1] if (_HERE.parents[1] / "configs").is_dir() else _HERE.parents[2]
