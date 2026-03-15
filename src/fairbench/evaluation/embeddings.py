"""Embedding evaluator for semantic similarity analysis."""

from typing import Any

from fairbench.core.exceptions import EvaluationError
from fairbench.core.types import GeneratedOutput
from fairbench.evaluation.base import Evaluator


class EmbeddingEvaluator(Evaluator):
    """Evaluator that computes text embeddings.

    Embeddings are used for:
    - Counterfactual Divergence Score (CDS) computation
    - Output diversity analysis
    - Semantic similarity comparisons
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        """Initialize the embedding evaluator.

        Args:
            model_name: Sentence transformer model to use.
            device: Device to run on ("cpu", "cuda", "mps").
            batch_size: Batch size for encoding.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None

    def _get_model(self) -> Any:
        """Lazily load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise EvaluationError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
        return self._model

    async def evaluate(self, output: GeneratedOutput) -> dict[str, Any]:
        """Compute embedding for a generated output.

        Args:
            output: The generated output.

        Returns:
            Dictionary with 'embedding' key containing the vector.
        """
        model = self._get_model()

        try:
            # Encode the text
            embedding = model.encode(
                output.text,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return {"embedding": embedding.tolist()}
        except Exception as e:
            raise EvaluationError(f"Embedding computation failed: {e}") from e

    async def evaluate_batch(
        self, outputs: list[GeneratedOutput]
    ) -> list[dict[str, Any]]:
        """Compute embeddings for multiple outputs efficiently.

        Args:
            outputs: List of generated outputs.

        Returns:
            List of dictionaries with embeddings.
        """
        if not outputs:
            return []

        model = self._get_model()

        try:
            texts = [output.text for output in outputs]
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=self.batch_size,
            )
            return [{"embedding": emb.tolist()} for emb in embeddings]
        except Exception as e:
            raise EvaluationError(f"Batch embedding computation failed: {e}") from e

    @property
    def name(self) -> str:
        """Get the evaluator name."""
        return "embeddings"

    def compute_similarity(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity score (0-1, higher = more similar).
        """
        import numpy as np

        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def compute_distance(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        """Compute cosine distance between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine distance (0-2, lower = more similar).
        """
        return 1.0 - self.compute_similarity(embedding1, embedding2)
