"""Validate scenario YAML files against the Chapter 6 schema.

    python ch06/validate.py ch06/examples/*.yaml
    python ch06/validate.py my_scenario.yaml

Exits non-zero if any file fails to validate, so it can gate a CI pipeline.
Pass --schema to print the JSON Schema instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scenario_schema import json_schema, load_scenario  # noqa: E402


def main(argv: list[str]) -> int:
    if "--schema" in argv:
        print(json.dumps(json_schema(), indent=2))
        return 0

    paths = [a for a in argv if not a.startswith("-")]
    if not paths:
        print("usage: python ch06/validate.py <scenario.yaml> [more.yaml ...]")
        return 2

    failures = 0
    for p in paths:
        try:
            s = load_scenario(p)
            print(f"OK    {p}  (id={s.id}, domain={s.domain}, harm_type={s.harm_type})")
        except Exception as e:  # noqa: BLE001 - report any validation error clearly
            failures += 1
            print(f"FAIL  {p}\n      {type(e).__name__}: {e}")

    print(f"\n{len(paths) - failures}/{len(paths)} valid")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
