"""OpenAI-compatible API adapter.

This adapter works with any API that implements the OpenAI chat completions
interface, including:
- Together AI
- Anyscale
- Groq
- Fireworks AI
- Local servers (vLLM, llama.cpp, etc.)
"""

import os
import time
from typing import Any

from fairbench.adapters.base import ModelAdapter
from fairbench.core.exceptions import AdapterError
from fairbench.core.types import GeneratedOutput, GenerationConfig, ModelInfo


class OpenAICompatibleAdapter(ModelAdapter):
    """Adapter for OpenAI-compatible APIs."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        max_tokens: int = 1024,
        timeout: int = 60,
        provider_name: str = "openai-compatible",
    ) -> None:
        """Initialize the OpenAI-compatible adapter.

        Args:
            model: Model name/ID for the API.
            base_url: Base URL for the API (e.g., "https://api.together.xyz/v1").
            api_key: API key. If not provided, tries api_key_env.
            api_key_env: Environment variable name for API key.
            max_tokens: Default max tokens for generation.
            timeout: Request timeout in seconds.
            provider_name: Name of the provider for logging/info.
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.provider_name = provider_name
        self._client: Any = None

        # Resolve API key
        if api_key:
            self.api_key = api_key
        elif api_key_env:
            self.api_key = os.environ.get(api_key_env)
        else:
            # Try common patterns
            self.api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY")

        # Some local servers don't need API keys
        # We'll only error if the server actually requires auth

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
                api_key=self.api_key or "not-needed",
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GeneratedOutput:
        """Generate text using an OpenAI-compatible API.

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
            raise AdapterError(f"{self.provider_name} API error: {e}") from e

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract text from response
        text = ""
        if response.choices:
            text = response.choices[0].message.content or ""

        token_count = None
        input_tokens = None
        if response.usage:
            token_count = response.usage.completion_tokens
            input_tokens = response.usage.prompt_tokens

        return GeneratedOutput(
            text=text,
            prompt=prompt,
            model_info=self.get_model_info(),
            generation_config=config,
            latency_ms=latency_ms,
            token_count=token_count,
            metadata={
                "input_tokens": input_tokens,
                "finish_reason": response.choices[0].finish_reason if response.choices else None,
                "base_url": self.base_url,
            },
        )

    def get_model_info(self) -> ModelInfo:
        """Get information about the model."""
        return ModelInfo(
            name=self.model,
            provider=self.provider_name,
            parameters={
                "max_tokens": self.max_tokens,
                "base_url": self.base_url,
            },
        )

    @property
    def name(self) -> str:
        """Get the adapter name."""
        return f"{self.provider_name}:{self.model}"
