"""Anthropic Claude model adapter."""

import os
import time
from typing import Any

from fairbench.adapters.base import ModelAdapter
from fairbench.core.exceptions import AdapterError
from fairbench.core.types import GeneratedOutput, GenerationConfig, ModelInfo


class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic Claude models."""

    DEFAULT_MODEL = "claude-sonnet-4-5"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
        timeout: int = 60,
    ) -> None:
        """Initialize the Anthropic adapter.

        Args:
            model: Model name (e.g., "claude-sonnet-4-20250514").
            api_key: API key. If not provided, uses ANTHROPIC_API_KEY env var.
            max_tokens: Default max tokens for generation.
            timeout: Request timeout in seconds.
        """
        self.model = model or self.DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client: Any = None

        if not self.api_key:
            raise AdapterError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key parameter."
            )

    def _get_client(self) -> Any:
        """Lazily initialize the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise AdapterError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
            self._client = anthropic.AsyncAnthropic(
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GeneratedOutput:
        """Generate text using Claude.

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
            # Anthropic does not allow temperature and top_p to both be set.
            # Use top_p only when explicitly requested (non-default value).
            create_kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": config.max_tokens or self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if config.top_p != 1.0:
                create_kwargs["top_p"] = config.top_p
            else:
                create_kwargs["temperature"] = config.temperature
            if config.stop_sequences:
                create_kwargs["stop_sequences"] = config.stop_sequences

            response = await client.messages.create(**create_kwargs)
        except Exception as e:
            raise AdapterError(f"Anthropic API error: {e}") from e

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract text from response
        text = ""
        if response.content:
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text

        return GeneratedOutput(
            text=text,
            prompt=prompt,
            model_info=self.get_model_info(),
            generation_config=config,
            latency_ms=latency_ms,
            token_count=response.usage.output_tokens if response.usage else None,
            metadata={
                "input_tokens": response.usage.input_tokens if response.usage else None,
                "stop_reason": response.stop_reason,
            },
        )

    def get_model_info(self) -> ModelInfo:
        """Get information about the Claude model."""
        return ModelInfo(
            name=self.model,
            provider="anthropic",
            parameters={
                "max_tokens": self.max_tokens,
            },
        )

    @property
    def name(self) -> str:
        """Get the adapter name."""
        return f"anthropic:{self.model}"

    def validate_config(self, config: GenerationConfig) -> GenerationConfig:
        """Validate configuration for Anthropic models."""
        # Anthropic requires max_tokens
        if config.max_tokens <= 0:
            return GenerationConfig(
                max_tokens=self.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                seed=config.seed,
                stop_sequences=config.stop_sequences,
            )
        return config
