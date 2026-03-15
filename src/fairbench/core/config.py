"""Configuration management for FAIRBench."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from fairbench.core.exceptions import ConfigError


class StorageConfig(BaseModel):
    """Storage backend configuration."""

    backend: str = "sqlite"
    sqlite_path: str = "~/.fairbench/fairbench.db"
    postgres_url: str | None = None

    model_config = {"frozen": True}


class EvaluatorConfig(BaseModel):
    """Configuration for an evaluator."""

    backend: str = "local"
    model: str | None = None
    threshold: float | None = None
    api_key: str | None = None

    model_config = {"frozen": True}


class EvaluationConfig(BaseModel):
    """Evaluation pipeline configuration."""

    concurrency: int = 10
    timeout_seconds: int = 60
    retry_attempts: int = 3
    evaluators: dict[str, EvaluatorConfig] = Field(default_factory=dict)

    model_config = {"frozen": True}


class MetricConfig(BaseModel):
    """Configuration for a specific metric."""

    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class MetricsConfig(BaseModel):
    """Metrics engine configuration."""

    default_baseline: str = "uniform"
    rsi: MetricConfig = Field(default_factory=MetricConfig)
    sar: MetricConfig = Field(default_factory=MetricConfig)
    ode: MetricConfig = Field(default_factory=MetricConfig)
    cds: MetricConfig = Field(default_factory=MetricConfig)
    hsi: MetricConfig = Field(default_factory=MetricConfig)

    model_config = {"frozen": True}


class ModelAdapterConfig(BaseModel):
    """Configuration for a model adapter."""

    api_key: str | None = None
    model: str
    base_url: str | None = None
    max_tokens: int = 1024
    timeout: int = 60

    model_config = {"frozen": True}


class ReportingConfig(BaseModel):
    """Reporting configuration."""

    output_dir: str = "./reports"
    formats: list[str] = Field(default_factory=lambda: ["json", "html"])
    include_raw_outputs: bool = False

    model_config = {"frozen": True}


class Config(BaseModel):
    """Main FAIRBench configuration."""

    version: str = "1.0"
    log_level: str = "INFO"
    storage: StorageConfig = Field(default_factory=StorageConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    models: dict[str, ModelAdapterConfig] = Field(default_factory=dict)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    model_config = {"frozen": True}


_config: Config | None = None


def load_config(path: Path | str | None = None) -> Config:
    """Load configuration from a YAML file.

    Args:
        path: Path to the config file. If None, looks for fairbench.yaml
              in the current directory, then ~/.fairbench/config.yaml.

    Returns:
        The loaded configuration.

    Raises:
        ConfigError: If the config file is invalid.
    """
    if path is None:
        # Look for config in standard locations
        candidates = [
            Path("fairbench.yaml"),
            Path("fairbench.yml"),
            Path.home() / ".fairbench" / "config.yaml",
            Path.home() / ".fairbench" / "config.yml",
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break

    if path is None:
        # No config file found, use defaults
        return Config()

    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in config file: {e}") from e

    if data is None:
        return Config()

    # Handle environment variable expansion
    data = _expand_env_vars(data)

    try:
        return Config.model_validate(data.get("fairbench", data))
    except Exception as e:
        raise ConfigError(f"Invalid configuration: {e}") from e


def _expand_env_vars(data: Any) -> Any:
    """Recursively expand environment variables in config values."""
    import os
    import re

    if isinstance(data, str):
        # Match ${VAR} or $VAR patterns
        pattern = r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)"

        def replace(match: re.Match[str]) -> str:
            var_name = match.group(1) or match.group(2)
            return os.environ.get(var_name, match.group(0))

        return re.sub(pattern, replace, data)
    elif isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    return data


def get_config() -> Config:
    """Get the current configuration, loading it if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration to force reload."""
    global _config
    _config = None
