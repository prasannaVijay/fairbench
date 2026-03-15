"""Custom exceptions for FAIRBench."""


class FairBenchError(Exception):
    """Base exception for all FAIRBench errors."""

    pass


class ConfigError(FairBenchError):
    """Error in configuration."""

    pass


class ScenarioError(FairBenchError):
    """Error loading or validating scenarios."""

    pass


class AdapterError(FairBenchError):
    """Error in model adapter."""

    pass


class EvaluationError(FairBenchError):
    """Error during evaluation."""

    pass


class MetricError(FairBenchError):
    """Error computing metrics."""

    pass


class StorageError(FairBenchError):
    """Error in storage backend."""

    pass


class BaselineNotFoundError(FairBenchError):
    """Requested baseline distribution not found."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Baseline distribution not found: {name}")
