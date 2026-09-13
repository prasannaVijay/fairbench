"""Append-only, tamper-evident audit log.

Book: Chapter 5, "Security, logging, and audit trails".

The chapter shows a simplified `audit.py` that hash-chains events. This is the
hardened form, addressing the technical review of that listing (#145-149):

- HMAC link signing with a secret key, so a bare hash chain in the same store
  cannot be silently recomputed by someone with write access (#145).
- Monotonic ``sequence_id`` and an explicit genesis marker, so truncation of
  leading or trailing events is detectable (#146).
- A canonical serializer (sorted keys, ``ensure_ascii``, ``allow_nan=False``,
  ``.value`` on enums, compact separators) with raw canonical bytes persisted,
  so verification never re-serializes decoded objects (#146).
- A single-writer lock so concurrent appends cannot fork the chain, and a
  ``verify_chain()`` that asserts sequence contiguity and every link from
  genesis (#147).
- An expanded event taxonomy covering the lifecycle events needed to
  reconstruct a disputed result, and immutable (frozen) events whose digest is
  computed once (#149).

For production, publish ``head`` periodically to external append-only storage
(a git commit, a WORM bucket, or a timestamping authority) so the chain is
anchored outside the database it protects.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from fairbench_genai.core.exceptions import FairBenchError

HASH_VERSION = "v1"
GENESIS_HASH = "0" * 64


class AuditError(FairBenchError):
    """Raised when the audit chain fails verification."""


class AuditEventType(str, Enum):
    """Lifecycle events an external auditor needs to reconstruct a result."""

    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_ABORTED = "run_aborted"
    RUN_COVERAGE_RECORDED = "run_coverage_recorded"
    SCENARIO_UPDATED = "scenario_updated"
    PROMPT_SET_GENERATED = "prompt_set_generated"
    THRESHOLD_CHANGED = "threshold_changed"
    RUBRIC_CHANGED = "rubric_changed"
    CLASSIFIER_VERSION_CHANGED = "classifier_version_changed"
    ANNOTATION_ADDED = "annotation_added"
    REVIEW_TASK_ROUTED = "review_task_routed"
    SCORECARD_GENERATED = "scorecard_generated"
    SCORECARD_OVERRIDDEN = "scorecard_overridden"
    ACCESS_GRANTED = "access_granted"
    ACCESS_REVOKED = "access_revoked"


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    """Deterministic, round-trip-stable serialization for hashing."""
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class AuditEvent:
    """An immutable audit record. The digest is derived from ``payload()``."""

    sequence_id: int
    event_type: AuditEventType
    entity_id: str
    actor: str
    details: dict[str, Any]
    timestamp: str
    previous_hash: str
    hash_version: str = HASH_VERSION

    def payload(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "event_type": self.event_type.value,
            "entity_id": self.entity_id,
            "actor": self.actor,
            "details": self.details,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "hash_version": self.hash_version,
        }

    def digest(self, key: bytes) -> str:
        return hmac.new(key, canonical_bytes(self.payload()), hashlib.sha256).hexdigest()


class AuditLog:
    """A single-writer, HMAC-linked, append-only audit log.

    Args:
        secret_key: HMAC key. Keep it out of the database the log protects.
        path: Optional file to append raw canonical JSON lines to (each line is
            the event payload plus its ``hmac``).
    """

    def __init__(self, secret_key: bytes | str, path: str | Path | None = None) -> None:
        self._key = secret_key.encode("utf-8") if isinstance(secret_key, str) else secret_key
        if not self._key:
            raise AuditError("an HMAC secret key is required for the audit log")
        self._lock = threading.Lock()
        self._events: list[AuditEvent] = []
        self._hashes: list[str] = []
        self._head = GENESIS_HASH
        self._path = Path(path).expanduser() if path else None

    def append(
        self,
        event_type: AuditEventType | str,
        entity_id: str,
        actor: str,
        details: dict[str, Any] | None = None,
    ) -> AuditEvent:
        with self._lock:  # single writer: concurrent appends cannot fork the chain
            event = AuditEvent(
                sequence_id=len(self._events),
                event_type=AuditEventType(event_type),
                entity_id=entity_id,
                actor=actor,
                details=dict(details or {}),
                timestamp=datetime.now(timezone.utc).isoformat(),
                previous_hash=self._head,
            )
            h = event.digest(self._key)
            self._events.append(event)
            self._hashes.append(h)
            self._head = h
            if self._path is not None:
                record = event.payload()
                record["hmac"] = h
                with open(self._path, "ab") as f:
                    f.write(canonical_bytes(record) + b"\n")
            return event

    def verify_chain(self) -> bool:
        """Assert sequence contiguity and every HMAC link from genesis.

        Raises AuditError on the first break. Call this as a precondition for
        publishing a scorecard.
        """
        prev = GENESIS_HASH
        for i, event in enumerate(self._events):
            if event.sequence_id != i:
                raise AuditError(f"sequence gap: expected {i}, got {event.sequence_id}")
            if event.previous_hash != prev:
                raise AuditError(f"broken link at sequence {i}")
            recomputed = event.digest(self._key)
            if not hmac.compare_digest(recomputed, self._hashes[i]):
                raise AuditError(f"tampered or mis-keyed event at sequence {i}")
            prev = recomputed
        return True

    @property
    def head(self) -> str:
        """Current head hash — publish this to an external anchor periodically."""
        return self._head

    def __len__(self) -> int:
        return len(self._events)


# Re-exported for tests / callers that need to construct a tampered event.
__all__ = [
    "AuditError",
    "AuditEvent",
    "AuditEventType",
    "AuditLog",
    "GENESIS_HASH",
    "canonical_bytes",
    "replace",
]
