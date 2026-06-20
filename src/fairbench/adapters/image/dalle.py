"""DALL-E 3 image generation adapter."""

import os
import time
from typing import Any

from fairbench.adapters.image.base import ImageModelAdapter
from fairbench.core.exceptions import AdapterError
from fairbench.core.image_types import GeneratedImage, ImageGenerationConfig
from fairbench.core.types import ModelInfo

# gpt-image-1 supported values
_GPT_IMAGE_SIZES = {"1024x1024", "1536x1024", "1024x1536", "auto"}
_GPT_IMAGE_QUALITIES = {"low", "medium", "high", "auto"}

# Legacy dall-e-3 supported values (kept for reference)
_DALLE3_SIZES = {"1024x1024", "1792x1024", "1024x1792"}
_DALLE3_QUALITIES = {"standard", "hd"}
_DALLE3_STYLES = {"vivid", "natural"}


class DALLEAdapter(ImageModelAdapter):
    """Adapter for OpenAI DALL-E 3 image generation.

    Uses the OpenAI Images API (images.generate endpoint).
    Images are returned as URLs and optionally downloaded to bytes.

    Usage:
        adapter = DALLEAdapter()
        image = await adapter.generate("A professional soccer player scoring a goal")
    """

    DEFAULT_MODEL = "gpt-image-1"

    def __init__(
        self,
        model: str = "gpt-image-1",
        api_key: str | None = None,
        timeout: int = 120,
        download_images: bool = True,
        save_dir: str | None = None,
    ) -> None:
        """Initialize the DALL-E adapter.

        Args:
            model: Model name. Currently only "dall-e-3" is supported.
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            timeout: Request timeout in seconds.
            download_images: If True, download image bytes from the returned URL.
            save_dir: Optional directory to save images to disk.
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout
        self.download_images = download_images
        self.save_dir = save_dir
        self._client: Any = None
        self._http_client: Any = None

        if not self.api_key:
            raise AdapterError(
                "OpenAI API key not provided. Set OPENAI_API_KEY env var or pass api_key."
            )

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise AdapterError("openai package not installed. Run: pip install openai")
            self._client = openai.AsyncOpenAI(api_key=self.api_key, timeout=self.timeout)
        return self._client

    async def _download_image(self, url: str) -> bytes:
        """Download image bytes from a URL."""
        try:
            import httpx
        except ImportError:
            raise AdapterError("httpx package not installed. Run: pip install httpx")

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content

    def validate_config(self, config: ImageGenerationConfig) -> ImageGenerationConfig:
        """Enforce gpt-image-1 constraints."""
        size = config.size if config.size in _GPT_IMAGE_SIZES else "1024x1024"
        quality = config.quality if config.quality in _GPT_IMAGE_QUALITIES else "auto"
        return ImageGenerationConfig(
            size=size,
            quality=quality,
            n=1,
            seed=config.seed,
        )

    async def generate(
        self,
        prompt: str,
        config: ImageGenerationConfig | None = None,
    ) -> GeneratedImage:
        """Generate an image using DALL-E 3.

        Args:
            prompt: Text description of the image to generate.
            config: Optional generation configuration.

        Returns:
            GeneratedImage with URL and optional bytes.

        Raises:
            AdapterError: If the API call fails or returns a content policy refusal.
        """
        config = config or ImageGenerationConfig()
        config = self.validate_config(config)
        client = self._get_client()

        start_time = time.perf_counter()

        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
                "size": config.size,
                "quality": config.quality,
                "n": 1,
            }
            if config.style:
                kwargs["style"] = config.style

            response = await client.images.generate(**kwargs)
        except Exception as e:
            err_str = str(e)
            # Content policy refusal comes back as a 400 with "content_policy_violation"
            if "content_policy" in err_str.lower() or "safety" in err_str.lower():
                latency_ms = (time.perf_counter() - start_time) * 1000
                return GeneratedImage(
                    prompt=prompt,
                    model_info=self.get_model_info(),
                    generation_config=config,
                    image_url=None,
                    latency_ms=latency_ms,
                    metadata={"refused": True, "refusal_reason": err_str},
                )
            raise AdapterError(f"DALL-E API error: {e}") from e

        latency_ms = (time.perf_counter() - start_time) * 1000
        image_item = response.data[0]
        image_url = getattr(image_item, "url", None)
        revised_prompt = getattr(image_item, "revised_prompt", None)

        # gpt-image-1 returns base64 data directly; legacy dall-e-3 returned URLs
        image_data: bytes | None = None
        b64 = getattr(image_item, "b64_json", None)
        if b64:
            import base64 as _b64
            image_data = _b64.b64decode(b64)
        elif self.download_images and image_url:
            try:
                image_data = await self._download_image(image_url)
            except Exception:
                pass  # URL still usable; don't fail the whole run

        # Optionally save to disk
        image_path = None
        if self.save_dir and image_data:
            import hashlib
            from pathlib import Path
            save_dir = Path(self.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            fname = hashlib.md5(prompt.encode()).hexdigest()[:16] + ".png"
            image_path = save_dir / fname
            image_path.write_bytes(image_data)

        return GeneratedImage(
            prompt=prompt,
            model_info=self.get_model_info(),
            generation_config=config,
            image_data=image_data,
            image_url=image_url,
            image_path=image_path,
            revised_prompt=revised_prompt,
            latency_ms=latency_ms,
            metadata={
                "model": self.model,
                "revised_prompt": revised_prompt,
            },
        )

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.model,
            provider="openai",
            parameters={"quality": "standard"},
        )

    @property
    def name(self) -> str:
        return f"dalle:{self.model}"
