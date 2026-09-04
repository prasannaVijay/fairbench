"""Test-time import paths and shared fixtures.

The package lives under ``src/`` and the per-chapter code lives in top-level
chapter directories whose modules import each other by bare name, exactly as the
book's listings do. Both go on ``sys.path`` here so that ``python -m pytest``
works from a clean checkout without an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for _path in (ROOT / "src", ROOT / "ch08"):
    _entry = str(_path)
    if _path.is_dir() and _entry not in sys.path:
        sys.path.append(_entry)

# The package directory takes precedence over anything a chapter directory
# happens to shadow.
_src = str(ROOT / "src")
if _src in sys.path:
    sys.path.remove(_src)
    sys.path.insert(0, _src)

import pytest  # noqa: E402  - imported after sys.path is arranged above

from fairbench_genai.scenarios.registry import reset_registry  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_scenario_registry():
    """Give each test a fresh global scenario registry.

    The registry is a module-level singleton; without this, two tests that load
    the same scenario file collide with 'already registered'.
    """
    reset_registry()
    yield
    reset_registry()
