"""Core type definitions for FAIRBench."""

from datetime import datetime, timezone


def _utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class FairnessDimension(str, Enum):
    """The four fairness dimensions in FAIRBench."""

    REPRESENTATIONAL = "representational"
    DISTRIBUTIONAL = "distributional"
    INTERACTIONAL = "interactional"
    PROCEDURAL = "procedural"


class SensitiveAttribute(str, Enum):
    """Common sensitive attributes for fairness testing."""

    GENDER = "gender"
    RACE = "race"
    ETHNICITY = "ethnicity"
    AGE = "age"
    RELIGION = "religion"
    NATIONALITY = "nationality"
    DISABILITY = "disability"
    SEXUAL_ORIENTATION = "sexual_orientation"
    SOCIOECONOMIC = "socioeconomic"


class BaselineType(str, Enum):
    """Types of baseline distributions for metrics."""

    UNIFORM = "uniform"
    REAL_WORLD = "real_world"
    CUSTOM = "custom"
    TRAINING_DATA = "training_data"


# --- Model-related types ---


class GenerationConfig(BaseModel):
    """Configuration for model generation."""

    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    seed: int | None = None
    stop_sequences: list[str] = Field(default_factory=list)

    model_config = {"frozen": True}


class ModelInfo(BaseModel):
    """Information about a model for reproducibility."""

    name: str
    provider: str
    version: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class GeneratedOutput(BaseModel):
    """Output from a generative model."""

    text: str
    prompt: str
    model_info: ModelInfo
    generation_config: GenerationConfig
    latency_ms: float | None = None
    token_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utc_now)

    model_config = {"frozen": True}


# --- Scenario types ---


class CounterfactualVariant(BaseModel):
    """A counterfactual variant of a prompt."""

    prompt: str
    attribute_value: str
    description: str | None = None

    model_config = {"frozen": True}


class CounterfactualGroup(BaseModel):
    """A group of counterfactual variants for a sensitive attribute."""

    attribute: SensitiveAttribute | str
    variants: list[CounterfactualVariant]

    model_config = {"frozen": True}


class Scenario(BaseModel):
    """A single fairness test scenario."""

    id: str
    prompt: str
    description: str | None = None
    dimensions: list[FairnessDimension] = Field(default_factory=list)
    counterfactuals: list[CounterfactualGroup] = Field(default_factory=list)
    expected_behavior: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class ScenarioSet(BaseModel):
    """A collection of related scenarios."""

    name: str
    version: str = "1.0"
    description: str | None = None
    dimensions: list[FairnessDimension] = Field(default_factory=list)
    scenarios: list[Scenario]
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


# --- Evaluation types ---


class ToxicityScores(BaseModel):
    """Toxicity evaluation scores."""

    toxicity: float = 0.0
    severe_toxicity: float = 0.0
    identity_attack: float = 0.0
    insult: float = 0.0
    threat: float = 0.0
    sexually_explicit: float = 0.0

    model_config = {"frozen": True}


class SentimentScores(BaseModel):
    """Sentiment analysis scores."""

    positive: float = 0.0
    negative: float = 0.0
    neutral: float = 0.0

    model_config = {"frozen": True}


class EvaluatedOutput(BaseModel):
    """A generated output with evaluation annotations."""

    id: UUID = Field(default_factory=uuid4)
    output: GeneratedOutput
    scenario_id: str
    is_counterfactual: bool = False
    counterfactual_attribute: str | None = None
    counterfactual_value: str | None = None
    original_prompt: str | None = None  # For counterfactuals, the base prompt

    # Evaluation results
    embedding: list[float] | None = None
    toxicity: ToxicityScores | None = None
    sentiment: SentimentScores | None = None
    detected_entities: dict[str, list[str]] = Field(default_factory=dict)
    custom_evaluations: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


# --- Metric types ---


class MetricResult(BaseModel):
    """Result from computing a fairness metric."""

    metric_name: str
    value: float
    std: float | None = None
    n_samples: int
    confidence_interval: tuple[float, float] | None = None
    interpretation: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utc_now)

    model_config = {"frozen": True}


class MetricThresholds(BaseModel):
    """Thresholds for metric interpretation."""

    good: float
    acceptable: float
    poor: float

    def interpret(self, value: float) -> str:
        """Interpret a metric value."""
        if value <= self.good:
            return "good"
        elif value <= self.acceptable:
            return "acceptable"
        else:
            return "poor"


# --- Run types ---


class RunStatus(str, Enum):
    """Status of an evaluation run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationRun(BaseModel):
    """A complete evaluation run."""

    id: UUID = Field(default_factory=uuid4)
    status: RunStatus = RunStatus.PENDING
    model_info: ModelInfo
    scenario_sets: list[str]
    metrics_requested: list[str]

    # Results
    outputs: list[EvaluatedOutput] = Field(default_factory=list)
    metric_results: list[MetricResult] = Field(default_factory=list)

    # Timing
    created_at: datetime = Field(default_factory=_utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Configuration snapshot
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None


# --- Distribution types (for baselines) ---


class Distribution(BaseModel):
    """A probability distribution over categories."""

    probabilities: dict[str, float]

    def __init__(self, probabilities: dict[str, float] | None = None, **kwargs: Any):
        if probabilities is None:
            probabilities = kwargs.get("probabilities", {})
        # Normalize probabilities
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in probabilities.items()}
        super().__init__(probabilities=probabilities)

    @classmethod
    def uniform(cls, categories: list[str] | None = None) -> "Distribution":
        """Create a uniform distribution."""
        if categories is None:
            categories = []
        n = len(categories)
        if n == 0:
            return cls(probabilities={})
        return cls(probabilities={cat: 1.0 / n for cat in categories})

    def get(self, key: str, default: float = 0.0) -> float:
        """Get probability for a category."""
        return self.probabilities.get(key, default)

    def categories(self) -> list[str]:
        """Get all categories."""
        return list(self.probabilities.keys())

    model_config = {"frozen": True}
