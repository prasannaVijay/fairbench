"""Evaluate a benchmark result against the Chapter 9 gate policy.

Book: Chapter 9, "Gates and guardrails".

    python ch09/gate_evaluator.py ch09/examples/result_soft_breach.yaml
    python ch09/gate_evaluator.py ch09/examples/result_hard_breach.yaml
    python ch09/gate_evaluator.py ch09/examples/result_clean.yaml
    python ch09/gate_evaluator.py --gates ch09/gates.yaml --exceptions ch09/exception_log.yaml \
        ch09/examples/result_soft_breach.yaml

Exits non-zero when the decision is ``block_deployment``, so a CI job can gate
on it directly. An escalation exits zero: a soft gate flags a result for human
review without blocking the deployment, and a pipeline that treated the two
identically would collapse the graded response the chapter is built around.

The evaluation has three steps:

1. Read each metric value in the result against the conditions in the gate
   policy. A hard-gate breach blocks. A soft-gate breach escalates to the named
   reviewer.
2. Apply any valid exception from the exception log. An unexpired exception
   naming a breached soft-gate metric turns that escalation into an accepted,
   documented deviation, carrying the approver and the expiry date with it.
   Nothing suppresses a hard gate.
3. Return one decision for the run: ``block_deployment`` if any hard gate
   fired, ``escalate_for_review`` if any soft gate fired without a covering
   exception, and ``pass`` otherwise.

A metric the gate policy names but the result does not carry is reported as
not evaluated. It is neither a pass nor a breach, and the summary says so,
because a silently missing metric is the failure mode that makes a green gate
meaningless.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exception_record import (  # noqa: E402
    ExceptionLog,
    ExceptionRecord,
    load_exception_log,
)

_HERE = Path(__file__).resolve().parent
DEFAULT_GATES_PATH = _HERE / "gates.yaml"

# The gate conditions the chapter prints are simple comparisons of a metric
# value against a constant: "value > 0.10", "value < 0.60". Parsing them with a
# regex keeps the policy file declarative and keeps arbitrary expressions out
# of the evaluator, which matters because this file decides whether a model
# ships.
_CONDITION = re.compile(r"^\s*value\s*(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)\s*$")

Decision = Literal["pass", "escalate_for_review", "block_deployment"]
GateStatus = Literal["pass", "breach", "breach_excepted", "not_evaluated"]


@dataclass(frozen=True)
class Gate:
    """One row of the gate policy: a metric, a condition, and a response."""

    metric: str
    condition: str
    response: str
    tier: Literal["hard", "soft"]
    reviewer: str | None = None

    def is_breached(self, value: float) -> bool:
        match = _CONDITION.match(self.condition)
        if not match:
            raise ValueError(
                f"{self.metric}: unsupported gate condition {self.condition!r}; "
                "expected the form 'value > 0.10' or 'value < 0.60'"
            )
        operator, threshold_text = match.groups()
        threshold = float(threshold_text)
        if operator == ">":
            return value > threshold
        if operator == ">=":
            return value >= threshold
        if operator == "<":
            return value < threshold
        return value <= threshold

    @property
    def threshold(self) -> float:
        match = _CONDITION.match(self.condition)
        if not match:
            raise ValueError(f"{self.metric}: unsupported gate condition {self.condition!r}")
        return float(match.group(2))


@dataclass(frozen=True)
class GateOutcome:
    """What happened at one gate for one run."""

    gate: Gate
    status: GateStatus
    value: float | None = None
    exception: ExceptionRecord | None = None

    def describe(self) -> str:
        if self.status == "not_evaluated":
            return f"{self.gate.metric:16s} not evaluated  (no value in result)"
        assert self.value is not None
        head = f"{self.gate.metric:16s} {self.value:<8.4f} [{self.gate.tier}] {self.gate.condition}"
        if self.status == "pass":
            return f"{head}  -> pass"
        if self.status == "breach_excepted":
            assert self.exception is not None
            return (
                f"{head}  -> breach, excepted by {self.exception.exception_id} "
                f"(approved by {self.exception.approved_by}, expires "
                f"{self.exception.expires_at:%Y-%m-%d})"
            )
        if self.gate.tier == "hard":
            return f"{head}  -> BREACH, {self.gate.response}"
        return f"{head}  -> breach, {self.gate.response} ({self.gate.reviewer})"


@dataclass
class GateReport:
    """The decision for a whole run, plus the per-gate detail behind it."""

    decision: Decision
    outcomes: list[GateOutcome] = field(default_factory=list)

    @property
    def blocking(self) -> list[GateOutcome]:
        return [o for o in self.outcomes if o.status == "breach" and o.gate.tier == "hard"]

    @property
    def escalations(self) -> list[GateOutcome]:
        return [o for o in self.outcomes if o.status == "breach" and o.gate.tier == "soft"]

    @property
    def excepted(self) -> list[GateOutcome]:
        return [o for o in self.outcomes if o.status == "breach_excepted"]

    @property
    def not_evaluated(self) -> list[GateOutcome]:
        return [o for o in self.outcomes if o.status == "not_evaluated"]

    @property
    def reviewers(self) -> list[str]:
        seen: list[str] = []
        for outcome in self.escalations:
            if outcome.gate.reviewer and outcome.gate.reviewer not in seen:
                seen.append(outcome.gate.reviewer)
        return seen

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable form, for the ``gate_decision`` column and MLflow."""
        return {
            "decision": self.decision,
            "blocked_on": [o.gate.metric for o in self.blocking],
            "escalated_on": [o.gate.metric for o in self.escalations],
            "excepted": {
                o.gate.metric: o.exception.exception_id for o in self.excepted if o.exception
            },
            "not_evaluated": [o.gate.metric for o in self.not_evaluated],
            "reviewers": self.reviewers,
        }

    def summary(self) -> str:
        lines = [o.describe() for o in self.outcomes]
        lines.append("")
        lines.append(f"decision: {self.decision}")
        if self.blocking:
            metrics = ", ".join(o.gate.metric for o in self.blocking)
            lines.append(f"  blocked on hard gate(s): {metrics}")
        if self.escalations:
            metrics = ", ".join(o.gate.metric for o in self.escalations)
            lines.append(f"  escalated on soft gate(s): {metrics}")
            lines.append(f"  reviewers: {', '.join(self.reviewers)}")
        if self.excepted:
            metrics = ", ".join(o.gate.metric for o in self.excepted)
            lines.append(f"  breached but covered by a live exception: {metrics}")
        if self.not_evaluated:
            metrics = ", ".join(o.gate.metric for o in self.not_evaluated)
            lines.append(f"  not evaluated (no value in result): {metrics}")
        return "\n".join(lines)


class GatePolicy:
    """The hard and soft gates from ``ch09/gates.yaml``."""

    def __init__(self, hard: list[Gate], soft: list[Gate]) -> None:
        self.hard = hard
        self.soft = soft

    @property
    def gates(self) -> list[Gate]:
        return [*self.hard, *self.soft]

    @property
    def hard_gate_metrics(self) -> tuple[str, ...]:
        return tuple(g.metric for g in self.hard)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_GATES_PATH) -> "GatePolicy":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict) or "gates" not in data:
            raise ValueError(f"{path}: missing top-level 'gates' key")
        body = data["gates"] or {}
        return cls(
            hard=[cls._gate(entry, "hard") for entry in body.get("hard") or []],
            soft=[cls._gate(entry, "soft") for entry in body.get("soft") or []],
        )

    @staticmethod
    def _gate(entry: dict[str, Any], tier: Literal["hard", "soft"]) -> Gate:
        missing = {"metric", "condition", "response"} - set(entry)
        if missing:
            raise ValueError(f"gate entry {entry!r} is missing {sorted(missing)}")
        if tier == "soft" and not entry.get("reviewer"):
            raise ValueError(
                f"{entry['metric']}: a soft gate must name a reviewer, otherwise the "
                "escalation it opens has nowhere to go"
            )
        return Gate(
            metric=entry["metric"],
            condition=entry["condition"],
            response=entry["response"],
            tier=tier,
            reviewer=entry.get("reviewer"),
        )

    def evaluate(
        self,
        metrics: dict[str, float | None],
        exceptions: ExceptionLog | None = None,
        at: datetime | None = None,
    ) -> GateReport:
        """Judge a set of metric values against this policy.

        Args:
            metrics: Metric name to value. A name the policy gates but this
                mapping omits (or maps to None) is reported as not evaluated.
            exceptions: A validated exception log. Only unexpired records
                covering a breached soft gate are applied.
            at: The moment to judge expiry against. Defaults to now, and is
                supplied by the tests so an expiry can be crossed on purpose.
        """
        at = at or datetime.now(timezone.utc)
        outcomes: list[GateOutcome] = []

        for gate in self.gates:
            value = metrics.get(gate.metric)
            if value is None:
                outcomes.append(GateOutcome(gate=gate, status="not_evaluated"))
                continue
            value = float(value)
            if not gate.is_breached(value):
                outcomes.append(GateOutcome(gate=gate, status="pass", value=value))
                continue
            # A hard gate is never suppressed, whatever the exception log says.
            record = (
                exceptions.active_for(gate.metric, at)
                if exceptions is not None and gate.tier == "soft"
                else None
            )
            if record is not None:
                outcomes.append(
                    GateOutcome(
                        gate=gate, status="breach_excepted", value=value, exception=record
                    )
                )
            else:
                outcomes.append(GateOutcome(gate=gate, status="breach", value=value))

        report = GateReport(decision="pass", outcomes=outcomes)
        if report.blocking:
            report.decision = "block_deployment"
        elif report.escalations:
            report.decision = "escalate_for_review"
        return report


def evaluate_result(
    metrics: dict[str, float | None],
    gates_path: str | Path = DEFAULT_GATES_PATH,
    exceptions_path: str | Path | None = None,
    at: datetime | None = None,
) -> GateReport:
    """Convenience entry point: load the policy and the log, then evaluate."""
    policy = GatePolicy.load(gates_path)
    log = None
    if exceptions_path is not None:
        log = load_exception_log(exceptions_path, policy.hard_gate_metrics)
    return policy.evaluate(metrics, log, at=at)


def load_metrics(path: str | Path) -> dict[str, float | None]:
    """Read a metrics result file (YAML or JSON, same shape).

    Accepts either a bare mapping of metric name to value, or a run record with
    the values under a ``metrics`` key, which is the shape ``fairbench.run``
    writes.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of metric values")
    values = data.get("metrics", data)
    if not isinstance(values, dict):
        raise ValueError(f"{path}: 'metrics' must be a mapping of metric name to value")
    return {str(k): (None if v is None else float(v)) for k, v in values.items()}


def main(argv: list[str]) -> int:
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        return 0

    gates_path: str | Path = DEFAULT_GATES_PATH
    exceptions_path: str | Path | None = None
    as_json = "--json" in argv

    args = [a for a in argv if a != "--json"]
    positional: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--gates":
            i += 1
            gates_path = args[i]
        elif arg == "--exceptions":
            i += 1
            exceptions_path = args[i]
        elif arg.startswith("-"):
            print(f"unknown option: {arg}")
            return 2
        else:
            positional.append(arg)
        i += 1

    if not positional:
        print(
            "usage: python ch09/gate_evaluator.py [--gates gates.yaml] "
            "[--exceptions exception_log.yaml] [--json] <result.yaml> [more.yaml ...]"
        )
        return 2

    worst = 0
    for path in positional:
        try:
            metrics = load_metrics(path)
            report = evaluate_result(metrics, gates_path, exceptions_path)
        except Exception as e:  # noqa: BLE001 - report any load/validation error clearly
            print(f"FAIL  {path}\n      {type(e).__name__}: {e}")
            worst = max(worst, 2)
            continue
        if as_json:
            print(json.dumps({"result": str(path), **report.to_dict()}, indent=2))
        else:
            print(f"--- {path}")
            print(report.summary())
            print()
        if report.decision == "block_deployment":
            worst = max(worst, 1)
    return worst


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
