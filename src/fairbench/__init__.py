"""FAIRBench: A fairness benchmarking framework for generative AI."""

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
