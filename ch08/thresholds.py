"""The threshold evaluation layer.

This is deliberately not part of the metric engine. The engine measures; this
layer judges. Keeping them apart is what lets the pilot hold its measure-first
stance and still hand a product owner a recommendation, and it means a change of
appetite for risk is a change to one YAML file rather than to the measurement.

Each metric is compared against the band structure in ``config/thresholds.yaml``
and the comparisons roll up into one verdict. A single red is enough to reach
``do_not_ship``, because the metrics answer different questions and a clean score
on one does not offset a failure on another.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

GREEN = "green"
AMBER = "amber"
RED = "red"


@dataclass(frozen=True)
class Band:
    """One metric's band structure, and which direction is good."""

    metric: str
    direction: str
    threshold: float
    red_at: float

    @property
    def higher_is_better(self) -> bool:
        return self.direction == "higher_is_better"

    def evaluate(self, value: float) -> str:
        if self.higher_is_better:
            if value >= self.threshold:
                return GREEN
            return AMBER if value >= self.red_at else RED
        if value <= self.threshold:
            return GREEN
        return AMBER if value <= self.red_at else RED


@dataclass(frozen=True)
class MetricVerdict:
    metric: str
    value: float
    band: str
    threshold: float
    direction: str


@dataclass(frozen=True)
class ThresholdVerdict:
    """The whole run's assessment: per-metric bands plus one recommendation."""

    summary: str
    metrics: list[MetricVerdict]
    flags: list[dict[str, str]]

    def band(self, metric: str) -> str:
        for m in self.metrics:
            if m.metric == metric:
                return m.band
        raise KeyError(metric)


DEFAULT_THRESHOLD_PATH = Path(__file__).resolve().parent / "config" / "thresholds.yaml"


class ThresholdEvaluator:
    """Compares metric values against the pilot's bands and rolls them up."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path else DEFAULT_THRESHOLD_PATH
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        self.config_path = Path(path)
        self.bands: dict[str, Band] = {
            name: Band(
                metric=name,
                direction=spec.get("direction", "lower_is_better"),
                threshold=float(spec["threshold"]),
                red_at=float(spec["red_at"]),
            )
            for name, spec in raw["metrics"].items()
        }
        self.verdict_rules: dict[str, str] = raw.get("verdict", {})
        self.flag_rules: list[dict[str, str]] = raw.get("flags", []) or []

    def metric_order(self) -> list[str]:
        """The order the summary prints metrics in, taken from the config file."""
        return list(self.bands)

    def evaluate(self, metrics: Mapping[str, Any]) -> ThresholdVerdict:
        verdicts: list[MetricVerdict] = []
        for name, band in self.bands.items():
            if name not in metrics:
                continue
            value = float(metrics[name])
            # The band is decided on the value as it is reported, that is,
            # rounded to two places, so the colour beside a number in the
            # summary can never disagree with the number itself.
            verdicts.append(
                MetricVerdict(
                    metric=name,
                    value=round(value, 2),
                    band=band.evaluate(round(value, 2)),
                    threshold=band.threshold,
                    direction=band.direction,
                )
            )

        bands = {v.band for v in verdicts}
        if RED in bands:
            summary = self.verdict_rules.get("any_red", "do_not_ship")
        elif AMBER in bands:
            summary = self.verdict_rules.get("any_amber", "ship_with_conditions")
        else:
            summary = self.verdict_rules.get("all_green", "ship")

        flags: list[dict[str, str]] = []
        by_metric = {v.metric: v for v in verdicts}
        for rule in self.flag_rules:
            verdict = by_metric.get(rule.get("metric", ""))
            if verdict is not None and verdict.band == rule.get("when"):
                flags.append({"metric": verdict.metric, "note": rule["note"]})

        return ThresholdVerdict(summary=summary, metrics=verdicts, flags=flags)


__all__ = [
    "AMBER",
    "Band",
    "GREEN",
    "MetricVerdict",
    "RED",
    "ThresholdEvaluator",
    "ThresholdVerdict",
]
