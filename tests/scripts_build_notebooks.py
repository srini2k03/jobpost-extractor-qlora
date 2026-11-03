"""Import shim so tests can read constants from scripts/build_notebooks.py."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_build_notebooks", Path(__file__).resolve().parents[1] / "scripts" / "build_notebooks.py"
)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
GPU_PINS = _mod.GPU_PINS
