#!/usr/bin/env python3
"""Chapter 5's offline soccer demo, reachable at the path the book prints.

Chapter 5's quickstart runs `python ch05/soccer_stub_benchmark.py`. In the
companion repository (`fairbench-book`) that file holds the demo itself; here
in the library repository the demo lives at `examples/soccer_stub_benchmark.py`
alongside the other runnable examples, because it exercises the library rather
than standing apart from it.

This wrapper exists so that the command printed in the book works in either
repository. It runs no API calls and needs no key: the stub adapter replays
recorded outputs through the same code paths a live run would take.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent / "examples" / "soccer_stub_benchmark.py"


def main() -> int:
    if not DEMO.exists():  # pragma: no cover - only reachable on a broken checkout
        print(f"could not find the demo at {DEMO}", file=sys.stderr)
        return 1
    runpy.run_path(str(DEMO), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
