"""Image generation model adapters."""

from fairbench.adapters.image.base import ImageModelAdapter
from fairbench.adapters.image.dalle import DALLEAdapter
from fairbench.adapters.image.stable_diffusion import StableDiffusionAdapter

__all__ = ["ImageModelAdapter", "DALLEAdapter", "StableDiffusionAdapter"]
