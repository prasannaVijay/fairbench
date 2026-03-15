"""Model adapters for FAIRBench."""

from fairbench.adapters.base import ModelAdapter
from fairbench.adapters.anthropic import AnthropicAdapter
from fairbench.adapters.openai import OpenAIAdapter
from fairbench.adapters.openai_compatible import OpenAICompatibleAdapter
from fairbench.adapters.http_webhook import HTTPWebhookAdapter
from fairbench.adapters.registry import AdapterRegistry, get_adapter_registry

__all__ = [
    "ModelAdapter",
    "AnthropicAdapter",
    "OpenAIAdapter",
    "OpenAICompatibleAdapter",
    "HTTPWebhookAdapter",
    "AdapterRegistry",
    "get_adapter_registry",
]
