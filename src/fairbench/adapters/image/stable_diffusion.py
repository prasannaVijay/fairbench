"""Stable Diffusion image generation adapter.

Supports two backends:
  - HuggingFace Inference API (hosted, requires HF_API_TOKEN)
  - Local diffusers pipeline (requires torch + diffusers installed)
"""

import io
import os
import time
from typing import Any

from fairbench.adapters.image.base import ImageModelAdapter
from fairbench.core.exceptions import AdapterError
from fairbench.core.image_types import GeneratedImage, ImageGenerationConfig
from fairbench.core.types import ModelInfo


class StableDiffusionAdapter(ImageModelAdapter):
    """Adapter for Stable Diffusion via HuggingFace Inference API or local diffusers.

    HuggingFace Inference API (default, no GPU required):
        adapter = StableDiffusionAdapter(
            model="stabilityai/stable-diffusion-xl-base-1.0",
            backend="hf_api",
            api_token="hf_...",
        )

    Local diffusers (requires torch + diffusers, GPU recommended):
        adapter = StableDiffusionAdapter(
            model="stabilityai/stable-diffusion-xl-base-1.0",
            backend="local",
            device="cuda",
        )
    """

    DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        backend: str = "hf_api",
        api_token: str | None = None,
        device: str = "cpu",
        save_dir: str | None = None,
        timeout: int = 120,
    ) -> None:
        """Initialize the Stable Diffusion adapter.

        Args:
            model: HuggingFace model ID.
            backend: "hf_api" (Inference API) or "local" (diffusers).
            api_token: HuggingFace token. Falls back to HF_API_TOKEN env var.
            device: Device for local backend ("cpu", "cuda", "mps").
            save_dir: Optional directory to persist generated images.
            timeout: Request timeout in seconds (hf_api only).
        """
        self.model = model
        self.backend = backend
        self.api_token = api_token or os.environ.get("HF_API_TOKEN")
        self.device = device
        self.save_dir = save_dir
        self.timeout = timeout
        self._pipeline: Any = None  # For local backend

        if backend == "hf_api" and not self.api_token:
            raise AdapterError(
                "HuggingFace API token not provided. "
                "Set HF_API_TOKEN env var or pass api_token."
            )

    async def generate(
        self,
        prompt: str,
        config: ImageGenerationConfig | None = None,
    ) -> GeneratedImage:
        """Generate an image using Stable Diffusion.

        Args:
            prompt: Text description of the desired image.
            config: Optional generation configuration.

        Returns:
            GeneratedImage with image_data (PNG bytes).
        """
        config = config or ImageGenerationConfig()

        if self.backend == "hf_api":
            return await self._generate_hf_api(prompt, config)
        elif self.backend == "local":
            return await self._generate_local(prompt, config)
        else:
            raise AdapterError(f"Unknown backend: {self.backend!r}. Use 'hf_api' or 'local'.")

    async def _generate_hf_api(
        self, prompt: str, config: ImageGenerationConfig
    ) -> GeneratedImage:
        """Generate via HuggingFace Inference API."""
        try:
            import httpx
        except ImportError:
            raise AdapterError("httpx not installed. Run: pip install httpx")

        api_url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {self.api_token}"}

        payload: dict[str, Any] = {"inputs": prompt}
        parameters: dict[str, Any] = {}
        if config.negative_prompt:
            parameters["negative_prompt"] = config.negative_prompt
        if config.num_inference_steps:
            parameters["num_inference_steps"] = config.num_inference_steps
        if config.guidance_scale:
            parameters["guidance_scale"] = config.guidance_scale
        if config.seed is not None:
            parameters["seed"] = config.seed
        if parameters:
            payload["parameters"] = parameters

        start_time = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(api_url, headers=headers, json=payload)
                response.raise_for_status()
                image_data = response.content
        except Exception as e:
            err_str = str(e)
            if "loading" in err_str.lower() or "503" in err_str:
                raise AdapterError(
                    f"HF model {self.model!r} is loading. "
                    "Wait 20–30 seconds and retry."
                ) from e
            raise AdapterError(f"HuggingFace API error: {e}") from e

        latency_ms = (time.perf_counter() - start_time) * 1000
        image_path = self._maybe_save(prompt, image_data)

        return GeneratedImage(
            prompt=prompt,
            model_info=self.get_model_info(),
            generation_config=config,
            image_data=image_data,
            image_path=image_path,
            latency_ms=latency_ms,
            metadata={"backend": "hf_api", "model": self.model},
        )

    async def _generate_local(
        self, prompt: str, config: ImageGenerationConfig
    ) -> GeneratedImage:
        """Generate using a local diffusers pipeline (runs in threadpool)."""
        import asyncio

        pipeline = self._get_local_pipeline()

        def _sync_generate() -> bytes:
            kwargs: dict[str, Any] = {"prompt": prompt}
            if config.negative_prompt:
                kwargs["negative_prompt"] = config.negative_prompt
            if config.num_inference_steps:
                kwargs["num_inference_steps"] = config.num_inference_steps
            if config.guidance_scale:
                kwargs["guidance_scale"] = config.guidance_scale
            if config.seed is not None:
                import torch
                kwargs["generator"] = torch.Generator(device=self.device).manual_seed(config.seed)

            result = pipeline(**kwargs)
            pil_image = result.images[0]
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            return buf.getvalue()

        start_time = time.perf_counter()
        loop = asyncio.get_event_loop()
        image_data = await loop.run_in_executor(None, _sync_generate)
        latency_ms = (time.perf_counter() - start_time) * 1000

        image_path = self._maybe_save(prompt, image_data)

        return GeneratedImage(
            prompt=prompt,
            model_info=self.get_model_info(),
            generation_config=config,
            image_data=image_data,
            image_path=image_path,
            latency_ms=latency_ms,
            metadata={"backend": "local", "device": self.device, "model": self.model},
        )

    def _get_local_pipeline(self) -> Any:
        if self._pipeline is None:
            try:
                import torch
                from diffusers import DiffusionPipeline
            except ImportError:
                raise AdapterError(
                    "diffusers and torch not installed. "
                    "Run: pip install diffusers transformers torch"
                )
            self._pipeline = DiffusionPipeline.from_pretrained(
                self.model,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            )
            self._pipeline = self._pipeline.to(self.device)
        return self._pipeline

    def _maybe_save(self, prompt: str, image_data: bytes) -> Any:
        if not self.save_dir:
            return None
        import hashlib
        from pathlib import Path
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = hashlib.md5(prompt.encode()).hexdigest()[:16] + ".png"
        image_path = save_dir / fname
        image_path.write_bytes(image_data)
        return image_path

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.model,
            provider="huggingface",
            parameters={"backend": self.backend, "device": self.device},
        )

    @property
    def name(self) -> str:
        short = self.model.split("/")[-1]
        return f"sd:{short}"
