"""Model adapters for FAIRBench."""

from fairbench_genai.adapters.base import ModelAdapter
from fairbench_genai.adapters.anthropic import AnthropicAdapter
from fairbench_genai.adapters.openai import OpenAIAdapter
from fairbench_genai.adapters.openai_compatible import OpenAICompatibleAdapter
from fairbench_genai.adapters.http_webhook import HTTPWebhookAdapter
from fairbench_genai.adapters.registry import AdapterRegistry, get_adapter_registry

__all__ = [
    "ModelAdapter",
    "AnthropicAdapter",
    "OpenAIAdapter",
    "OpenAICompatibleAdapter",
    "HTTPWebhookAdapter",
    "AdapterRegistry",
    "get_adapter_registry",
]
