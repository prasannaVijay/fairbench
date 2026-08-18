"""Tests for the Chapter 7 limitation-record schema."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch07"))
from limitation_record import (  # noqa: E402
    Limitation,
    MetricLimitationRecord,
    ScorecardLimitations,
    Severity,
    json_schema,
    load_limitations,
)

_EXAMPLE = Path(__file__).resolve().parent.parent / "ch07" / "examples" / "soccer_limitations.yaml"


def test_example_validates() -> None:
    sl = load_limitations(_EXAMPLE)
    assert sl.scorecard_id
    assert sl.compositional_risks
    metrics = {r.metric for r in sl.records}
    assert {"RSI_skin_tone", "DSI"} <= metrics
    rsi = next(r for r in sl.records if r.metric == "RSI_skin_tone")
    assert rsi.confidence_interval == (0.24, 0.39)
    assert rsi.human_review_coverage.inter_rater_reliability == 0.61
    # DSI records a false-negative refusal estimate (#97)
    dsi = next(r for r in sl.records if r.metric == "DSI")
    assert dsi.human_review_coverage.estimated_false_negative_rate == 0.06


def _base_record(**over) -> dict:
    d = dict(
        metric="X", construct="c", formula="f", aggregation="a",
        sample_size=10, value=0.1, owner="o", tracking_id="T", version="1.0",
    )
    d.update(over)
    return d


def test_directional_bias_requires_direction_and_bound() -> None:
    with pytest.raises(Exception):
        MetricLimitationRecord.model_validate(_base_record(
            limitations=[{"description": "d", "severity": "directional_bias"}]
        ))


def test_inference_blocking_requires_suppression() -> None:
    with pytest.raises(Exception):
        MetricLimitationRecord.model_validate(_base_record(
            limitations=[{"description": "prompt risk unmatched across arms", "severity": "inference_blocking"}],
            suppressed=False,
        ))
    # allowed when suppressed
    r = MetricLimitationRecord.model_validate(_base_record(
        limitations=[{"description": "prompt risk unmatched across arms", "severity": "inference_blocking"}],
        suppressed=True,
    ))
    assert r.suppressed is True


def test_unknown_field_rejected() -> None:
    with pytest.raises(Exception):
        MetricLimitationRecord.model_validate(_base_record(does_not_measure="everything else"))


def test_json_schema_exports() -> None:
    schema = json_schema()
    assert schema["title"] == "ScorecardLimitations"
