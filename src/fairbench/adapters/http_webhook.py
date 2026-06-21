"""Generic HTTP webhook adapter for any REST API.

This adapter allows users to connect to any model API by configuring
request/response templates in YAML, without writing Python code.
"""

import os
import re
import time
from typing import Any

import httpx

from fairbench.adapters.base import ModelAdapter
from fairbench.core.exceptions import AdapterError
from fairbench.core.types import GeneratedOutput, GenerationConfig, ModelInfo


class HTTPWebhookAdapter(ModelAdapter):
    """Generic adapter for any HTTP-based model API.

    Configure via YAML:

    ```yaml
    models:
      my-custom-model:
        type: http
        endpoint: "https://my-model.com/generate"
        method: POST
        headers:
          Authorization: "Bearer ${MY_TOKEN}"
          Content-Type: "application/json"
        request_template: |
          {"prompt": "{{prompt}}", "max_tokens": {{max_tokens}}}
        response_path: "$.output.text"
    ```
    """

    def __init__(
        self,
        endpoint: str,
        method: str = "POST",
        headers: dict[str, str] | None = None,
        request_template: str | None = None,
        response_path: str = "$.text",
        timeout: int = 60,
        model_name: str = "http-webhook",
        include_raw_response: bool = False,
    ) -> None:
        """Initialize the HTTP webhook adapter.

        Args:
            endpoint: The API endpoint URL.
            method: HTTP method (POST, GET, etc.).
            headers: HTTP headers to include.
            request_template: Template for the request body (uses {{var}} syntax).
            response_path: JSONPath to extract response text.
            timeout: Request timeout in seconds.
            model_name: Name to use for this model in reports.
            include_raw_response: If True, store the full API response in
                metadata["raw_response"]. Disabled by default — raw responses
                may contain sensitive server data that would be persisted to
                the SQLite run store and included in JSON exports.
        """
        self.endpoint = endpoint
        self.method = method.upper()
        self.headers = self._expand_env_vars(headers or {})
        self.request_template = request_template or '{"prompt": "{{prompt}}"}'
        self.response_path = response_path
        self.timeout = timeout
        self.model_name = model_name
        self.include_raw_response = include_raw_response
        self._client: httpx.AsyncClient | None = None

    def _expand_env_vars(self, data: dict[str, str]) -> dict[str, str]:
        """Expand environment variables in header values."""
        result = {}
        for key, value in data.items():
            # Match ${VAR} or $VAR patterns
            pattern = r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)"

            def replace(match: re.Match[str]) -> str:
                var_name = match.group(1) or match.group(2)
                return os.environ.get(var_name, match.group(0))

            result[key] = re.sub(pattern, replace, value)
        return result

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _render_template(
        self, template: str, prompt: str, config: GenerationConfig
    ) -> str:
        """Render the request template with variables.

        Args:
            template: The template string.
            prompt: The prompt to insert.
            config: Generation configuration.

        Returns:
            Rendered template string.
        """
        # Escape the prompt for JSON
        escaped_prompt = prompt.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

        result = template
        result = result.replace("{{prompt}}", escaped_prompt)
        result = result.replace("{{max_tokens}}", str(config.max_tokens))
        result = result.replace("{{temperature}}", str(config.temperature))
        result = result.replace("{{top_p}}", str(config.top_p))

        if config.seed is not None:
            result = result.replace("{{seed}}", str(config.seed))
        else:
            result = result.replace("{{seed}}", "null")

        return result

    def _extract_response(self, data: Any, path: str) -> str:
        """Extract text from response using JSONPath-like syntax.

        Args:
            data: The response data.
            path: Path to extract (e.g., "$.output.text" or "output.text").

        Returns:
            Extracted text.
        """
        # Simple JSONPath implementation
        path = path.lstrip("$").lstrip(".")

        current = data
        for part in path.split("."):
            if not part:
                continue

            # Handle array indexing
            array_match = re.match(r"(\w+)\[(\d+)\]", part)
            if array_match:
                key, index = array_match.groups()
                if isinstance(current, dict):
                    current = current.get(key, [])
                if isinstance(current, list) and len(current) > int(index):
                    current = current[int(index)]
                else:
                    return ""
            elif isinstance(current, dict):
                current = current.get(part, "")
            else:
                return ""

        return str(current) if current else ""

    async def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GeneratedOutput:
        """Generate text using the HTTP API.

        Args:
            prompt: The input prompt.
            config: Optional generation configuration.

        Returns:
            The generated output.

        Raises:
            AdapterError: If generation fails.
        """
        config = config or GenerationConfig()
        client = self._get_client()

        start_time = time.perf_counter()

        try:
            # Render request body
            body = self._render_template(self.request_template, prompt, config)

            # Make request
            if self.method == "POST":
                response = await client.post(
                    self.endpoint,
                    headers=self.headers,
                    content=body,
                )
            elif self.method == "GET":
                response = await client.get(
                    self.endpoint,
                    headers=self.headers,
                    params={"prompt": prompt},
                )
            else:
                raise AdapterError(f"Unsupported HTTP method: {self.method}")

            response.raise_for_status()
            data = response.json()

        except httpx.HTTPStatusError as e:
            raise AdapterError(
                f"HTTP error {e.response.status_code}: {e.response.text}"
            ) from e
        except Exception as e:
            raise AdapterError(f"HTTP request failed: {e}") from e

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract text from response
        text = self._extract_response(data, self.response_path)

        metadata: dict = {"endpoint": self.endpoint}
        if self.include_raw_response:
            metadata["raw_response"] = data

        return GeneratedOutput(
            text=text,
            prompt=prompt,
            model_info=self.get_model_info(),
            generation_config=config,
            latency_ms=latency_ms,
            metadata=metadata,
        )

    def get_model_info(self) -> ModelInfo:
        """Get information about the model."""
        return ModelInfo(
            name=self.model_name,
            provider="http-webhook",
            parameters={
                "endpoint": self.endpoint,
                "method": self.method,
            },
        )

    @property
    def name(self) -> str:
        """Get the adapter name."""
        return f"http:{self.model_name}"

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
