"""Base metric interface."""

from abc import ABC, abstractmethod

from fairbench.core.types import Distribution, EvaluatedOutput, MetricResult


class Metric(ABC):
    """Abstract base class for fairness metrics.

    Metrics compute quantitative measures of fairness from evaluated outputs.
    Each metric maps to one or more of the four fairness dimensions:
    - Representational
    - Distributional
    - Interactional
    - Procedural
    """

    @abstractmethod
    def compute(
        self,
        outputs: list[EvaluatedOutput],
        baseline: Distribution | None = None,
    ) -> MetricResult:
        """Compute the metric from evaluated outputs.

        Args:
            outputs: List of evaluated outputs to analyze.
            baseline: Optional baseline distribution for comparison.

        Returns:
            The computed metric result.
        """
        pass

    @abstractmethod
    def interpret(self, result: MetricResult) -> str:
        """Generate a human-readable interpretation of the result.

        Args:
            result: The metric result to interpret.

        Returns:
            A description of what the result means.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the metric name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get a description of what this metric measures."""
        pass

    @property
    def higher_is_better(self) -> bool:
        """Whether higher values indicate better fairness.

        Most fairness metrics are "lower is better" (less bias).
        Override in subclasses where higher is better.
        """
        return False

    def get_thresholds(self) -> dict[str, float]:
        """Get interpretation thresholds for this metric.

        Returns:
            Dictionary with 'good', 'acceptable', 'poor' thresholds.
        """
        return {
            "good": 0.1,
            "acceptable": 0.3,
            "poor": 0.5,
        }
