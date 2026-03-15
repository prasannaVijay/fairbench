"""Sentiment evaluator for tone analysis."""

from typing import Any

from fairbench.core.exceptions import EvaluationError
from fairbench.core.types import GeneratedOutput, SentimentScores
from fairbench.evaluation.base import Evaluator


class SentimentEvaluator(Evaluator):
    """Evaluator that analyzes sentiment of generated text.

    Sentiment analysis helps detect:
    - Differential tone across counterfactuals
    - Negative portrayals of certain groups
    - Emotional bias in responses
    """

    DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the sentiment evaluator.

        Args:
            model_name: Sentiment model to use.
            device: Device for inference.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self._classifier: Any = None

    def _get_classifier(self) -> Any:
        """Lazily load the sentiment classifier."""
        if self._classifier is None:
            try:
                from transformers import pipeline
            except ImportError:
                raise EvaluationError(
                    "transformers not installed. Run: pip install transformers"
                )

            self._classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=self.device if self.device != "cpu" else -1,
                top_k=None,  # Return all scores
            )
        return self._classifier

    async def evaluate(self, output: GeneratedOutput) -> dict[str, Any]:
        """Evaluate sentiment of a generated output.

        Args:
            output: The generated output.

        Returns:
            Dictionary with 'sentiment' key containing SentimentScores.
        """
        classifier = self._get_classifier()

        try:
            # Truncate text for model limits
            text = output.text[:512]

            # Get predictions
            results = classifier(text)

            # Parse results into SentimentScores
            scores = self._parse_results(results)

            return {"sentiment": scores}
        except Exception as e:
            raise EvaluationError(f"Sentiment evaluation failed: {e}") from e

    def _parse_results(self, results: list[Any]) -> SentimentScores:
        """Parse classifier results into SentimentScores.

        Args:
            results: Raw classifier output.

        Returns:
            Parsed SentimentScores.
        """
        scores = {
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 0.0,
        }

        if not results:
            return SentimentScores(**scores)

        # Handle list format
        if isinstance(results, list) and results:
            # Some models return nested list
            if isinstance(results[0], list):
                results = results[0]

            for item in results:
                if isinstance(item, dict):
                    label = item.get("label", "").lower()
                    score = item.get("score", 0.0)

                    # Map various label formats
                    if label in ("positive", "pos", "label_2"):
                        scores["positive"] = score
                    elif label in ("negative", "neg", "label_0"):
                        scores["negative"] = score
                    elif label in ("neutral", "neu", "label_1"):
                        scores["neutral"] = score

        return SentimentScores(**scores)

    async def evaluate_batch(
        self, outputs: list[GeneratedOutput]
    ) -> list[dict[str, Any]]:
        """Evaluate sentiment for multiple outputs.

        Args:
            outputs: List of generated outputs.

        Returns:
            List of dictionaries with sentiment scores.
        """
        if not outputs:
            return []

        classifier = self._get_classifier()

        try:
            texts = [output.text[:512] for output in outputs]
            results = classifier(texts)

            return [
                {"sentiment": self._parse_results([result])}
                for result in results
            ]
        except Exception as e:
            raise EvaluationError(f"Batch sentiment evaluation failed: {e}") from e

    @property
    def name(self) -> str:
        """Get the evaluator name."""
        return "sentiment"

    def get_dominant_sentiment(self, scores: SentimentScores) -> str:
        """Get the dominant sentiment label.

        Args:
            scores: The sentiment scores.

        Returns:
            "positive", "negative", or "neutral".
        """
        max_score = max(scores.positive, scores.negative, scores.neutral)
        if max_score == scores.positive:
            return "positive"
        elif max_score == scores.negative:
            return "negative"
        return "neutral"
