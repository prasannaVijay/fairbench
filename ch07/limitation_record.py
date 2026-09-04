"""A validatable schema for fairness-metric limitation records.

Book: Chapter 7, "The honest reckoning" / limitation documentation.

The chapter shows a limitation-record template. The technical review (#90-#97)
found it under-specified, and this module is the hardened form:

- Each record anchors to a POSITIVE construct: what the metric measures, its
  formula, aggregation, sample size, value, and confidence interval. A
  limitation is then an explicit deviation from that construct, not undefined
  "negative space" (#91).
- Known classifier error is propagated, not filed as a passive caveat: records
  carry ``classifier_accuracy`` and a data-suppression flag, and a
  ``directional_bias`` limitation must state its direction and bound (#92).
- Every limitation carries a severity grade; an ``inference_blocking`` finding
  halts automated publication until it is resolved (#96).
- Human-review coverage is structured numbers (sample %, design, agreement,
  inter-rater reliability, cell-level disagreement), not a prose string (#95).
- Records carry audit metadata (owner, tracking_id, version, amendment_log) and
  ``comparability_bounds``; a top-level block captures compositional risks (#90).
- Fields are uniform across metrics, so completeness can be audited.
"""

from __future__ import annotations

import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Severity(str, Enum):
    """How much a limitation degrades the metric."""

    SCOPE_NARROWING = "scope_narrowing"       # a deliberate, documented boundary
    INTERVAL_WIDENING = "interval_widening"   # adds uncertainty, widens the CI
    DIRECTIONAL_BIAS = "directional_bias"     # biases the point estimate one way
    INFERENCE_BLOCKING = "inference_blocking"  # invalidates the metric's inference


class HumanReviewCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_pct: float
    sampling_design: Literal["random", "stratified"]
    classifier_human_agreement_pct: float | None = None
    inter_rater_reliability: float | None = None      # kappa / alpha value
    irr_metric: str | None = None                     # e.g. "cohen_kappa", "krippendorff_alpha"
    # For DSI: sample of classified NON-refusals reviewed to estimate the
    # false-negative refusal rate the classifier missed (#97).
    non_refusal_review_pct: float | None = None
    estimated_false_negative_rate: float | None = None
    cell_disagreement: dict[str, float] | None = None


class Limitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    severity: Severity
    # Required when severity is directional_bias (#92):
    bias_direction: str | None = None   # e.g. "understates darker-tone representation"
    bias_bound: str | None = None       # e.g. "4-11 points"

    @model_validator(mode="after")
    def _directional_needs_bound(self) -> "Limitation":
        if self.severity == Severity.DIRECTIONAL_BIAS and not (self.bias_direction and self.bias_bound):
            raise ValueError("directional_bias limitations must state bias_direction and bias_bound")
        return self


class ComparabilityBounds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Conditions under which this score may be compared across runs / versions.
    comparable_across_runs_if: list[str] = Field(default_factory=list)


# ``construct`` is the field name the chapter prints, and the YAML records in a
# scorecard use it, so the schema keeps it. Pydantic warns that it shadows
# ``BaseModel.construct``, a deprecated classmethod this schema never calls; the
# warning is suppressed for this one class definition rather than globally, so a
# reader running the validator sees its output and nothing else.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r'Field name "construct" .* shadows an attribute in parent "BaseModel"',
        category=UserWarning,
    )

    class MetricLimitationRecord(BaseModel):
        model_config = ConfigDict(extra="forbid")

        metric: str
        # Positive construct (#91):
        construct: str
        formula: str
        aggregation: str
        sample_size: int
        value: float
        confidence_interval: tuple[float, float] | None = None
        # Error propagation / suppression (#92):
        classifier: str | None = None
        classifier_accuracy: float | None = None
        suppressed: bool = False              # insufficient_measurement_validity
        # Structured coverage (#95):
        human_review_coverage: HumanReviewCoverage | None = None
        # Cross-run comparability (#90):
        comparability_bounds: ComparabilityBounds | None = None
        # Limitations as explicit, severity-graded deviations (#96):
        limitations: list[Limitation] = Field(default_factory=list)
        # Audit metadata (#90):
        owner: str
        tracking_id: str
        version: str
        amendment_log: list[str] = Field(default_factory=list)

        @model_validator(mode="after")
        def _suppression_rule(self) -> "MetricLimitationRecord":
            blocking = any(limit.severity == Severity.INFERENCE_BLOCKING for limit in self.limitations)
            if blocking and not self.suppressed:
                raise ValueError(
                    f"{self.metric}: an inference_blocking limitation requires suppressed=true "
                    "(insufficient_measurement_validity) — the metric may not be published as a point estimate"
                )
            return self


class ScorecardLimitations(BaseModel):
    """The scorecard-level limitation block: compositional risks plus per-metric records."""

    model_config = ConfigDict(extra="forbid")

    scorecard_id: str
    compositional_risks: list[str] = Field(default_factory=list)  # e.g. averaging unequal-validity metrics
    records: list[MetricLimitationRecord]


def load_limitations(path: str | Path) -> ScorecardLimitations:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ScorecardLimitations.model_validate(data)


def json_schema() -> dict[str, Any]:
    return ScorecardLimitations.model_json_schema()
