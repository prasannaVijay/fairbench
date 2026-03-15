"""FAIRBench: A fairness benchmarking framework for generative AI."""

from fairbench.reporting.scorecard import generate_scorecard
from fairbench.core.engine import FairBenchEngine
from fairbench.core.types import (
    EvaluatedOutput,
    EvaluationRun,
    GeneratedOutput,
    GenerationConfig,
    MetricResult,
    ModelInfo,
    Scenario,
    ScenarioSet,
)

__version__ = "0.1.0"

__all__ = [
    "generate_scorecard",
    "FairBenchEngine",
    "EvaluatedOutput",
    "EvaluationRun",
    "GeneratedOutput",
    "GenerationConfig",
    "MetricResult",
    "ModelInfo",
    "Scenario",
    "ScenarioSet",
]
