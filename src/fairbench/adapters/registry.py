"""Adapter registry for managing model adapters."""

from typing import Any

from fairbench.adapters.base import ModelAdapter
from fairbench.core.config import ModelAdapterConfig
from fairbench.core.exceptions import AdapterError


class AdapterRegistry:
    """Registry for model adapters.

    The registry allows:
    - Registering custom adapters
    - Creating adapters from configuration
    - Looking up adapters by name
    """

    def __init__(self) -> None:
        self._adapters: dict[str, ModelAdapter] = {}
        self._factories: dict[str, type[ModelAdapter]] = {}
        self._register_builtin_factories()

    def _register_builtin_factories(self) -> None:
        """Register built-in adapter factories."""
        from fairbench.adapters.anthropic import AnthropicAdapter
        from fairbench.adapters.http_webhook import HTTPWebhookAdapter
        from fairbench.adapters.openai import OpenAIAdapter
        from fairbench.adapters.openai_compatible import OpenAICompatibleAdapter

        self._factories["anthropic"] = AnthropicAdapter
        self._factories["openai"] = OpenAIAdapter
        self._factories["openai-compatible"] = OpenAICompatibleAdapter
        self._factories["http"] = HTTPWebhookAdapter

    def register(self, name: str, adapter: ModelAdapter) -> None:
        """Register an adapter instance.

        Args:
            name: Name to register the adapter under.
            adapter: The adapter instance.
        """
        self._adapters[name] = adapter

    def register_factory(self, name: str, factory: type[ModelAdapter]) -> None:
        """Register an adapter factory (class).

        Args:
            name: Name for the adapter type.
            factory: The adapter class.
        """
        self._factories[name] = factory

    def get(self, name: str) -> ModelAdapter:
        """Get an adapter by name.

        Args:
            name: The adapter name.

        Returns:
            The adapter instance.

        Raises:
            AdapterError: If the adapter is not found.
        """
        if name not in self._adapters:
            raise AdapterError(f"Adapter not found: {name}")
        return self._adapters[name]

    def create(
        self,
        adapter_type: str,
        config: ModelAdapterConfig | dict[str, Any],
    ) -> ModelAdapter:
        """Create an adapter from configuration.

        Args:
            adapter_type: Type of adapter ("anthropic", "openai", etc.).
            config: Configuration for the adapter.

        Returns:
            The created adapter instance.

        Raises:
            AdapterError: If the adapter type is unknown.
        """
        if adapter_type not in self._factories:
            raise AdapterError(f"Unknown adapter type: {adapter_type}")

        factory = self._factories[adapter_type]

        # Convert config to dict if needed
        if isinstance(config, ModelAdapterConfig):
            config_dict = config.model_dump(exclude_none=True)
        else:
            config_dict = config

        try:
            return factory(**config_dict)
        except TypeError as e:
            raise AdapterError(
                f"Invalid configuration for {adapter_type} adapter: {e}"
            ) from e

    def create_and_register(
        self,
        name: str,
        adapter_type: str,
        config: ModelAdapterConfig | dict[str, Any],
    ) -> ModelAdapter:
        """Create an adapter and register it.

        Args:
            name: Name to register the adapter under.
            adapter_type: Type of adapter.
            config: Configuration for the adapter.

        Returns:
            The created and registered adapter.
        """
        adapter = self.create(adapter_type, config)
        self.register(name, adapter)
        return adapter

    def list_adapters(self) -> list[str]:
        """List registered adapter names."""
        return list(self._adapters.keys())

    def list_types(self) -> list[str]:
        """List available adapter types."""
        return list(self._factories.keys())

    def has(self, name: str) -> bool:
        """Check if an adapter is registered."""
        return name in self._adapters


# Global registry instance
_registry: AdapterRegistry | None = None


def get_adapter_registry() -> AdapterRegistry:
    """Get the global adapter registry."""
    global _registry
    if _registry is None:
        _registry = AdapterRegistry()
    return _registry


def reset_adapter_registry() -> None:
    """Reset the global adapter registry."""
    global _registry
    _registry = None
