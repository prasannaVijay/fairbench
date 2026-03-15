"""Base evaluator interface."""

from abc import ABC, abstractmethod
from typing import Any

from fairbench.core.types import EvaluatedOutput, GeneratedOutput


class Evaluator(ABC):
    """Abstract base class for output evaluators.

    Evaluators analyze generated outputs and add evaluation annotations
    (embeddings, toxicity scores, sentiment, etc.).
    """

    @abstractmethod
    async def evaluate(self, output: GeneratedOutput) -> dict[str, Any]:
        """Evaluate a generated output.

        Args:
            output: The generated output to evaluate.

        Returns:
            Dictionary of evaluation results to merge into EvaluatedOutput.
        """
        pass

    async def evaluate_batch(
        self, outputs: list[GeneratedOutput]
    ) -> list[dict[str, Any]]:
        """Evaluate multiple outputs.

        Default implementation calls evaluate() for each output.
        Subclasses can override for more efficient batch processing.

        Args:
            outputs: List of generated outputs.

        Returns:
            List of evaluation result dictionaries.
        """
        results = []
        for output in outputs:
            result = await self.evaluate(output)
            results.append(result)
        return results

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the evaluator name."""
        pass

    def applies_to(self, output: GeneratedOutput) -> bool:
        """Check if this evaluator applies to a given output.

        Subclasses can override to filter outputs (e.g., only text).

        Args:
            output: The output to check.

        Returns:
            True if this evaluator should process the output.
        """
        return True
