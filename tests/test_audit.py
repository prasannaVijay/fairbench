"""Tests for the Chapter 5 tamper-evident audit log."""

from dataclasses import replace

import pytest

from fairbench_genai.core.audit import (
    GENESIS_HASH,
    AuditError,
    AuditEventType,
    AuditLog,
)


def _log() -> AuditLog:
    log = AuditLog(secret_key="test-secret")
    log.append(AuditEventType.RUN_STARTED, entity_id="run1", actor="svc")
    log.append(AuditEventType.THRESHOLD_CHANGED, entity_id="run1", actor="alice",
               details={"metric": "RSI", "old": 0.25, "new": 0.15})
    log.append(AuditEventType.SCORECARD_GENERATED, entity_id="run1", actor="svc")
    return log


def test_chain_verifies() -> None:
    log = _log()
    assert len(log) == 3
    assert log.verify_chain() is True


def test_genesis_and_sequence() -> None:
    log = AuditLog(secret_key="k")
    e0 = log.append(AuditEventType.RUN_STARTED, "r", "svc")
    assert e0.sequence_id == 0
    assert e0.previous_hash == GENESIS_HASH
    e1 = log.append(AuditEventType.RUN_COMPLETED, "r", "svc")
    assert e1.sequence_id == 1
    assert e1.previous_hash == log._hashes[0]


def test_tampering_is_detected() -> None:
    log = _log()
    # Replace a stored event with an altered copy (frozen dataclass -> new object).
    log._events[1] = replace(log._events[1], details={"metric": "RSI", "old": 0.25, "new": 0.99})
    with pytest.raises(AuditError):
        log.verify_chain()


def test_wrong_key_fails_verification() -> None:
    log = _log()
    log._key = b"different-key"  # simulate a verifier without the real secret
    with pytest.raises(AuditError):
        log.verify_chain()


def test_missing_key_rejected() -> None:
    with pytest.raises(AuditError):
        AuditLog(secret_key="")


def test_file_persistence(tmp_path) -> None:
    p = tmp_path / "audit.log"
    log = AuditLog(secret_key="k", path=p)
    log.append(AuditEventType.RUN_STARTED, "r", "svc")
    log.append(AuditEventType.RUN_COMPLETED, "r", "svc")
    lines = p.read_bytes().decode().strip().splitlines()
    assert len(lines) == 2
    assert '"hmac":' in lines[0]
