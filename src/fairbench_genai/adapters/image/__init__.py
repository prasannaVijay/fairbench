"""Image generation model adapters."""

from fairbench_genai.adapters.image.base import ImageModelAdapter
from fairbench_genai.adapters.image.dalle import DALLEAdapter
from fairbench_genai.adapters.image.stable_diffusion import StableDiffusionAdapter
from fairbench_genai.adapters.image.stub import StubImageAdapter

__all__ = ["ImageModelAdapter", "DALLEAdapter", "StableDiffusionAdapter", "StubImageAdapter"]
