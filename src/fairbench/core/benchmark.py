"""Benchmark spec — the unified YAML format for describing a fairness audit.

A benchmark spec ties together the model under test, the scenarios to run,
the metrics to compute, and where to write the output.  It is the single
file a user authors to drive a complete fairness evaluation.

Example
-------
benchmark:
  name: "GPT-4o Gender & Occupation Audit"
  description: "Focused audit on gender bias in professional contexts"

model_under_test:
  provider: openai
  model: gpt-4o
  max_tokens: 1024
  temperature: 0.7

scenarios:
  - gender_occupation
  - racial_sentiment
  - ./my_custom_scenarios.yaml

metrics:
  - RSI
  - SAR
  - CDS
  - HSI
  - DSI

output:
  path: ./reports
  format: all        # json | md | all
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from fairbench.core.exceptions import ConfigError


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class BenchmarkMeta(BaseModel):
    """Top-level benchmark identity."""

    name: str = "FAIRBench Audit"
    description: str | None = None

    model_config = {"frozen": True}


class ModelUnderTest(BaseModel):
    """The model to evaluate."""

    provider: str  # anthropic | openai | openai_compatible | http_webhook
    model: str
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    api_key: str | None = None  # supports ${ENV_VAR} expansion
    base_url: str | None = None  # for openai_compatible / http_webhook

    model_config = {"frozen": True}


class OutputSpec(BaseModel):
    """Where and how to write the scorecard."""

    path: str = "./reports"
    format: str = "all"  # json | md | all

    model_config = {"frozen": True}


class BenchmarkSpec(BaseModel):
    """The complete benchmark specification parsed from a YAML file."""

    benchmark: BenchmarkMeta = Field(default_factory=BenchmarkMeta)
    model_under_test: ModelUnderTest
    scenarios: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(
        default_factory=lambda: ["RSI", "SAR", "ODE", "CDS", "HSI", "DSI"]
    )
    output: OutputSpec = Field(default_factory=OutputSpec)

    # The file this spec was loaded from — set by load_benchmark_spec()
    source_path: Path | None = Field(default=None, exclude=True)

    model_config = {"frozen": True}

    def resolve_scenario_paths(self) -> list[str]:
        """Return scenario entries with relative file paths resolved
        against the spec's source directory.

        Built-in names (no path separators, no file extension) are returned
        unchanged.  Relative file paths are resolved against the directory
        that contains the benchmark YAML.
        """
        base_dir = self.source_path.parent if self.source_path else Path(".")
        resolved: list[str] = []
        for entry in self.scenarios:
            p = Path(entry)
            # If it looks like a file path (has a suffix or an explicit ./ prefix)
            if p.suffix in (".yaml", ".yml", ".json") or entry.startswith(("./", "../")):
                resolved.append(str((base_dir / p).resolve()))
            else:
                resolved.append(entry)
        return resolved


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def is_benchmark_spec(path: Path | str) -> bool:
    """Return True if the YAML file at *path* is a benchmark spec rather
    than a plain scenario set.

    Detection is based on the presence of a ``model_under_test`` key at the
    top level of the document — this is the minimal distinguishing marker.
    """
    path = Path(path)
    if not path.exists():
        return False
    try:
        with open(path) as fh:
            data = yaml.safe_load(fh)
        return isinstance(data, dict) and "model_under_test" in data
    except Exception:
        return False


def load_benchmark_spec(path: Path | str) -> BenchmarkSpec:
    """Parse and validate a benchmark spec YAML file.

    Args:
        path: Path to the benchmark spec YAML.

    Returns:
        A validated :class:`BenchmarkSpec`.

    Raises:
        ConfigError: If the file is missing, malformed, or fails validation.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise ConfigError(f"Benchmark spec file not found: {path}")

    try:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in benchmark spec: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Benchmark spec must be a YAML mapping, got {type(raw)}")

    if "model_under_test" not in raw:
        raise ConfigError(
            "Benchmark spec must contain a 'model_under_test' key.  "
            "If this is a scenario set, pass it with --model to specify the model."
        )

    # Expand environment variables throughout the document
    raw = _expand_env_vars(raw)

    try:
        spec = BenchmarkSpec.model_validate(raw)
    except Exception as exc:
        raise ConfigError(f"Invalid benchmark spec: {exc}") from exc

    # Attach the source path (mutable workaround for frozen model)
    object.__setattr__(spec, "source_path", path)
    return spec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expand_env_vars(data: Any) -> Any:
    """Recursively expand ``${VAR}`` and ``$VAR`` in string values."""
    if isinstance(data, str):
        pattern = r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)"

        def _replace(m: re.Match[str]) -> str:
            var = m.group(1) or m.group(2)
            return os.environ.get(var, m.group(0))

        return re.sub(pattern, _replace, data)
    if isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    return data
