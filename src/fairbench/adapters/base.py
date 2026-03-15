"""Base model adapter interface."""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from fairbench.core.types import GeneratedOutput, GenerationConfig, ModelInfo


class ModelAdapter(ABC):
    """Abstract base class for model adapters.

    Model adapters provide a unified interface for interacting with
    different generative AI models, regardless of their underlying API.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GeneratedOutput:
        """Generate text from a prompt.

        Args:
            prompt: The input prompt.
            config: Optional generation configuration.

        Returns:
            The generated output.

        Raises:
            AdapterError: If generation fails.
        """
        pass

    async def generate_batch(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
    ) -> list[GeneratedOutput]:
        """Generate text from multiple prompts.

        Default implementation calls generate() for each prompt.
        Subclasses can override for more efficient batch processing.

        Args:
            prompts: List of input prompts.
            config: Optional generation configuration.

        Returns:
            List of generated outputs.
        """
        outputs = []
        for prompt in prompts:
            output = await self.generate(prompt, config)
            outputs.append(output)
        return outputs

    async def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        """Stream generated text from a prompt.

        Default implementation falls back to non-streaming generate().
        Subclasses can override for true streaming support.

        Args:
            prompt: The input prompt.
            config: Optional generation configuration.

        Yields:
            Text chunks as they are generated.
        """
        output = await self.generate(prompt, config)
        yield output.text

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Get information about the model.

        Returns:
            Model information for reproducibility.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the adapter name."""
        pass

    def validate_config(self, config: GenerationConfig) -> GenerationConfig:
        """Validate and adjust generation config for this adapter.

        Subclasses can override to enforce model-specific constraints.

        Args:
            config: The generation configuration.

        Returns:
            Validated/adjusted configuration.
        """
        return config
