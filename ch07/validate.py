"""Validate fairness-metric limitation records against the Chapter 7 schema.

    python ch07/validate.py ch07/examples/*.yaml
    python ch07/validate.py --schema

Exits non-zero if any file fails to validate, so it can gate a CI pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from limitation_record import json_schema, load_limitations  # noqa: E402


def main(argv: list[str]) -> int:
    if "--schema" in argv:
        print(json.dumps(json_schema(), indent=2))
        return 0
    paths = [a for a in argv if not a.startswith("-")]
    if not paths:
        print("usage: python ch07/validate.py <limitations.yaml> [more.yaml ...]")
        return 2
    failures = 0
    for p in paths:
        try:
            sl = load_limitations(p)
            metrics = ", ".join(r.metric for r in sl.records)
            print(f"OK    {p}  (scorecard={sl.scorecard_id}, records=[{metrics}])")
        except Exception as e:  # noqa: BLE001 - report any validation error clearly
            failures += 1
            print(f"FAIL  {p}\n      {type(e).__name__}: {e}")
    print(f"\n{len(paths) - failures}/{len(paths)} valid")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
