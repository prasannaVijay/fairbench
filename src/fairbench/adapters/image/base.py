"""Base image generation adapter interface."""

from abc import ABC, abstractmethod

from fairbench.core.image_types import GeneratedImage, ImageGenerationConfig
from fairbench.core.types import ModelInfo


class ImageModelAdapter(ABC):
    """Abstract base class for image generation model adapters.

    Mirrors ModelAdapter but produces GeneratedImage instead of GeneratedOutput.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: ImageGenerationConfig | None = None,
    ) -> GeneratedImage:
        """Generate an image from a text prompt.

        Args:
            prompt: Descriptive text prompt for image generation.
            config: Optional generation configuration.

        Returns:
            The generated image with metadata.

        Raises:
            AdapterError: If generation fails or is refused.
        """
        pass

    async def generate_batch(
        self,
        prompts: list[str],
        config: ImageGenerationConfig | None = None,
    ) -> list[GeneratedImage]:
        """Generate images from multiple prompts.

        Default implementation calls generate() sequentially.
        Subclasses can override for parallel generation.
        """
        results = []
        for prompt in prompts:
            image = await self.generate(prompt, config)
            results.append(image)
        return results

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Return model metadata for reproducibility."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique adapter name (e.g., 'dalle3', 'sd-xl')."""
        pass

    def validate_config(self, config: ImageGenerationConfig) -> ImageGenerationConfig:
        """Validate and adjust config for this adapter's constraints.

        Subclasses can override to enforce model-specific limits.
        """
        return config
