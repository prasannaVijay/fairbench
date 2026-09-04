"""Generate the root requirements.txt from pyproject.toml.

The Chapter 9 workflow installs dependencies with ``pip install -r
requirements.txt``, so the repository needs that file. It is generated from the
``[project].dependencies`` table in pyproject.toml rather than maintained by
hand, because a second hand-edited copy of a dependency list drifts from the
first one and then the CI job installs something the library does not.

    python ch09/tools/sync_requirements.py            # write requirements.txt
    python ch09/tools/sync_requirements.py --check    # verify it is in step

``--check`` exits non-zero when the generated content differs from the file on
disk, which is what tests/test_ch09_gates.py asserts.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
REQUIREMENTS = REPO_ROOT / "requirements.txt"

HEADER = """\
# requirements.txt
#
# Generated from the [project].dependencies table in pyproject.toml. Do not edit
# by hand: run `python ch09/tools/sync_requirements.py` instead, and
# `python ch09/tools/sync_requirements.py --check` to verify the two are in step
# (tests/test_ch09_gates.py runs that check).
#
# The Chapter 9 CI workflow installs from this file, which is why it exists at
# the repository root. It mirrors the dependency list rather than forking it, so
# pyproject.toml stays the one place a dependency is declared.
"""

FOOTER = """\

# The library itself, installed from the checkout, so that `python -m fairbench.run`
# can import fairbench_genai after actions/checkout.
-e .
"""


def render() -> str:
    with open(PYPROJECT, "rb") as f:
        data = tomllib.load(f)
    dependencies = data["project"]["dependencies"]
    return HEADER + "\n" + "\n".join(dependencies) + "\n" + FOOTER


def main(argv: list[str]) -> int:
    content = render()
    if "--check" in argv:
        if not REQUIREMENTS.exists():
            print(f"missing {REQUIREMENTS}")
            return 1
        current = REQUIREMENTS.read_text(encoding="utf-8")
        if current != content:
            print(
                f"{REQUIREMENTS} is out of step with pyproject.toml; "
                "run: python ch09/tools/sync_requirements.py"
            )
            return 1
        print(f"OK    {REQUIREMENTS} matches pyproject.toml")
        return 0
    REQUIREMENTS.write_text(content, encoding="utf-8")
    print(f"wrote {REQUIREMENTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
