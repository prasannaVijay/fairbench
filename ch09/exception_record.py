"""A validatable schema for fairness gate exception records.

Book: Chapter 9, "The exception process".

The chapter prints a filled exception log and states two rules about it in
passing. This module makes both rules executable, because an exception process
that is only described is the exception process the chapter warns about: a gate
with no enforced pathway gets routed around under deadline pressure, and the
governance record degrades into a list of the times a gate was bypassed.

The two rules:

- **No exception may name a hard gate.** The chapter says so in an inline
  comment on the ``gate_type`` field ("the evaluator rejects an exception log
  for any hard gate"). A hard gate exists precisely because its condition is
  not a matter of organizational tolerance, so the only resolution path is to
  fix the problem and rerun the benchmark. ``gate_type: hard`` fails
  validation, and so does a ``gate_type: soft`` record that names a metric the
  gate policy classifies as hard.
- **Expiry is enforced, not decorative.** ``expires_at`` must be later than
  ``approved_at``, and :meth:`ExceptionRecord.is_active` reports False once the
  date has passed. A lapsed exception is indistinguishable from no exception at
  all, so the gate fires again on the next run.

The remaining fields carry the three components the chapter names for a
well-designed exception process: a named approver who accepts accountability, a
structured justification, and a time to resolve by.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# The hard-gate metrics from ch09/gates.yaml. Kept here as a fallback so that a
# record can be validated on its own, without loading the gate policy; when the
# policy is available, ch09/gate_evaluator.py passes the real hard-gate list in.
DEFAULT_HARD_GATE_METRICS = ("HSI", "DSI")


def _as_aware(value: datetime) -> datetime:
    """Treat a naive timestamp as UTC, so comparisons never raise."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


class ExceptionRecord(BaseModel):
    """One governance decision to deploy despite a fired soft gate."""

    model_config = ConfigDict(extra="forbid")

    exception_id: str
    run_id: str
    metric: str
    # the evaluator rejects an exception log for any hard gate
    gate_type: Literal["soft"]
    metric_value: float
    threshold: float

    # Named approver: a person with the authority to authorize deployment
    # despite a fairness flag, who accepts accountability for that decision.
    approved_by: str
    approved_at: datetime
    # Expiry: the date by which the exception lapses and the deployment must
    # either have resolved the underlying issue or renewed the exception.
    expires_at: datetime

    # Documented justification: what the flag means, what the risk is, and why
    # deployment is appropriate despite it.
    justification: str
    conditions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _expiry_after_approval(self) -> "ExceptionRecord":
        if _as_aware(self.expires_at) <= _as_aware(self.approved_at):
            raise ValueError(
                f"{self.exception_id}: expires_at must be later than approved_at; "
                "an exception that expires when it is granted records nothing"
            )
        return self

    @model_validator(mode="after")
    def _justification_is_substantive(self) -> "ExceptionRecord":
        if len(self.justification.strip()) < 20:
            raise ValueError(
                f"{self.exception_id}: justification is too short to be an audit record"
            )
        return self

    def is_active(self, at: datetime | None = None) -> bool:
        """Whether this exception still suppresses its gate at the given time.

        Once ``expires_at`` has passed the exception stops applying, and the
        gate fires on the next benchmark run exactly as it did the first time.
        """
        now = _as_aware(at or datetime.now(timezone.utc))
        return now < _as_aware(self.expires_at)

    def applies_to(self, metric: str, at: datetime | None = None) -> bool:
        """Whether this exception covers a breach of ``metric`` right now."""
        return self.metric == metric and self.is_active(at)


class ExceptionLog(BaseModel):
    """The container the chapter's YAML uses: a top-level ``exception_log`` key.

    The book prints a single record under that key. A real log accumulates
    many, so ``exception_log`` accepts either one mapping or a list of them and
    normalizes to a list.
    """

    model_config = ConfigDict(extra="forbid")

    records: list[ExceptionRecord]

    def active_for(self, metric: str, at: datetime | None = None) -> ExceptionRecord | None:
        """The first unexpired exception covering ``metric``, if there is one."""
        for record in self.records:
            if record.applies_to(metric, at):
                return record
        return None

    def check_against_policy(self, hard_gate_metrics: tuple[str, ...] | list[str]) -> None:
        """Reject any record that names a metric the gate policy treats as hard.

        ``gate_type`` is already constrained to ``soft`` by the schema, which
        catches a record that admits what it is. This catches the more likely
        mistake: a record labelled ``soft`` that names HSI or DSI anyway.
        """
        hard = set(hard_gate_metrics)
        for record in self.records:
            if record.metric in hard:
                raise ValueError(
                    f"{record.exception_id}: {record.metric} is a hard gate; "
                    "a hard gate has no exception pathway. The only resolution "
                    "is to fix the breach and rerun the benchmark."
                )


def load_exception_log(
    path: str | Path,
    hard_gate_metrics: tuple[str, ...] | list[str] = DEFAULT_HARD_GATE_METRICS,
) -> ExceptionLog:
    """Load and validate an exception log from YAML.

    Raises ValueError (or a pydantic ValidationError) if the log names a hard
    gate, is missing a required field, or expires before it was approved.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    log = parse_exception_log(data)
    log.check_against_policy(hard_gate_metrics)
    return log


def parse_exception_log(data: Any) -> ExceptionLog:
    """Normalize the chapter's YAML shape into an :class:`ExceptionLog`."""
    if not isinstance(data, dict):
        raise ValueError("an exception log must be a mapping with an 'exception_log' key")
    if "exception_log" not in data:
        raise ValueError("missing top-level 'exception_log' key")
    body = data["exception_log"]
    entries = body if isinstance(body, list) else [body]
    return ExceptionLog(records=[ExceptionRecord.model_validate(e) for e in entries])


def json_schema() -> dict[str, Any]:
    return ExceptionLog.model_json_schema()
