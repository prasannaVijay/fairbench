"""OpenAI model adapter."""

import os
import time
from typing import Any

from fairbench.adapters.base import ModelAdapter
from fairbench.core.exceptions import AdapterError
from fairbench.core.types import GeneratedOutput, GenerationConfig, ModelInfo


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI models."""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
        timeout: int = 60,
        organization: str | None = None,
    ) -> None:
        """Initialize the OpenAI adapter.

        Args:
            model: Model name (e.g., "gpt-4o", "gpt-4o-mini").
            api_key: API key. If not provided, uses OPENAI_API_KEY env var.
            max_tokens: Default max tokens for generation.
            timeout: Request timeout in seconds.
            organization: Optional organization ID.
        """
        self.model = model or self.DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.organization = organization
        self._client: Any = None

        if not self.api_key:
            raise AdapterError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )

    def _get_client(self) -> Any:
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise AdapterError(
                    "openai package not installed. Run: pip install openai"
                )
            self._client = openai.AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization,
                timeout=self.timeout,
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GeneratedOutput:
        """Generate text using OpenAI models.

        Args:
            prompt: The input prompt.
            config: Optional generation configuration.

        Returns:
            The generated output.

        Raises:
            AdapterError: If generation fails.
        """
        config = config or GenerationConfig()
        config = self.validate_config(config)
        client = self._get_client()

        start_time = time.perf_counter()

        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": config.temperature,
                "top_p": config.top_p,
            }

            if config.max_tokens > 0:
                kwargs["max_tokens"] = config.max_tokens

            if config.stop_sequences:
                kwargs["stop"] = config.stop_sequences

            if config.seed is not None:
                kwargs["seed"] = config.seed

            response = await client.chat.completions.create(**kwargs)
        except Exception as e:
            raise AdapterError(f"OpenAI API error: {e}") from e

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract text from response
        text = ""
        if response.choices:
            text = response.choices[0].message.content or ""

        return GeneratedOutput(
            text=text,
            prompt=prompt,
            model_info=self.get_model_info(),
            generation_config=config,
            latency_ms=latency_ms,
            token_count=response.usage.completion_tokens if response.usage else None,
            metadata={
                "input_tokens": response.usage.prompt_tokens if response.usage else None,
                "finish_reason": response.choices[0].finish_reason if response.choices else None,
                "system_fingerprint": response.system_fingerprint,
            },
        )

    def get_model_info(self) -> ModelInfo:
        """Get information about the OpenAI model."""
        return ModelInfo(
            name=self.model,
            provider="openai",
            parameters={
                "max_tokens": self.max_tokens,
            },
        )

    @property
    def name(self) -> str:
        """Get the adapter name."""
        return f"openai:{self.model}"
