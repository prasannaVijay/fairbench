"""Typed schema and loader for a single benchmark run configuration.

Book: Chapter 5, "Run configuration".

The chapter shows a ``run_config.yaml`` as a simplified excerpt. This module is
the hardened, validatable form of it, addressing the technical review of that
listing:

- Retry taxonomy by HTTP status: transient statuses are retried, content-policy
  rejections fail fast and are recorded as a distinct outcome (#87, #95).
- Budget pre-flight and interleaving: reject an over-budget run before any API
  call, and keep sampling balanced across scenarios if a run aborts (#88).
- Rate limiting: a token bucket is the authoritative throttle; concurrency is a
  ceiling, and image-model timeouts are generous (#89).
- Run provenance: scenario-config hash, output path, checkpoint dir, schema
  version, and response-metadata capture (#90).
- ``capture_revised_prompt`` records the provider's rewritten prompt as a
  first-class signal (#86), and ``api_version`` maps to the API header version
  rather than a dated model snapshot, with ``seed`` documented as unsupported by
  many hosted image APIs (#91).

Loading a malformed run config raises ``ConfigError`` at load time instead of
surfacing as a confusing error deep inside execution.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fairbench_genai.core.exceptions import ConfigError


class RunOutcome(str, Enum):
    """Terminal outcome recorded for each attempt (a refusal is data, not an error)."""

    SUCCESS = "success"
    REFUSED_CONTENT_POLICY = "refused_content_policy"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"
    INTERNAL_ERROR = "internal_error"


class RetryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_attempts: int = 3
    backoff_type: Literal["fixed", "exponential"] = "exponential"
    backoff_base_seconds: float = 2.0
    # Transient statuses worth retrying, vs. terminal ones recorded as findings.
    retry_on_status: list[int] = Field(default_factory=lambda: [429, 500, 502, 503])
    fail_fast_on_status: list[int] = Field(default_factory=lambda: [400])

    @model_validator(mode="after")
    def _statuses_disjoint(self) -> "RetryPolicy":
        overlap = set(self.retry_on_status) & set(self.fail_fast_on_status)
        if overlap:
            raise ValueError(
                f"status codes cannot be both retried and fail-fast: {sorted(overlap)}"
            )
        return self


class RateLimit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rpm: int = 50
    strategy: Literal["token_bucket"] = "token_bucket"  # authoritative throttle


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concurrency: int = 10           # ceiling; the rate limit is authoritative
    timeout_seconds: int = 90       # image models generate slowly (15-30s)
    rate_limit: RateLimit = Field(default_factory=RateLimit)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replicates_per_prompt: int = 5
    seed: int | None = None         # many hosted image APIs do not support a seed


class BudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_cost_usd: float = 50.0
    abort_on_budget_exceed: bool = True
    preflight: bool = True              # reject an over-budget run before executing
    interleave_scenarios: bool = True   # keep sampling balanced if a run aborts
    cost_per_call_usd: float = 0.04     # used by the pre-flight estimate


class ModelSpec(BaseModel):
    # `model_id` and `api_version` use the model_ prefix; disable the protected
    # namespace so pydantic does not warn about shadowing.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    provider: str
    model_id: str
    api_version: str | None = None       # API header version, not a dated snapshot
    capture_revised_prompt: bool = True  # record the provider's rewritten prompt
    parameters: dict[str, Any] = Field(default_factory=dict)


class RunMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    scenario_ids: list[str]


class ProvenanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    scenario_config_hash: str | None = None
    output_path: str | None = None
    checkpoint_dir: str | None = None
    response_metadata: list[str] = Field(
        default_factory=lambda: ["request_id", "latency_ms", "finish_reason"]
    )


class RunConfig(BaseModel):
    # The top-level `model` block clashes with pydantic's protected namespace;
    # disable it so the natural YAML key `model:` maps to this field.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    run: RunMeta
    model: ModelSpec
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    provenance: ProvenanceConfig = Field(default_factory=ProvenanceConfig)

    def estimate_calls(self, prompts_per_scenario: int) -> int:
        """Total API calls this run would make (scenarios x prompts x replicates)."""
        return (
            len(self.run.scenario_ids)
            * prompts_per_scenario
            * self.sampling.replicates_per_prompt
        )

    def estimate_cost_usd(self, prompts_per_scenario: int) -> float:
        return self.estimate_calls(prompts_per_scenario) * self.budget.cost_per_call_usd

    def preflight(self, prompts_per_scenario: int) -> None:
        """Raise ConfigError if the estimated cost exceeds the budget.

        Call this before dispatching any request, so an over-budget run is
        rejected up front rather than aborting partway and truncating later
        scenarios.
        """
        if not self.budget.preflight:
            return
        est = self.estimate_cost_usd(prompts_per_scenario)
        if est > self.budget.max_cost_usd:
            raise ConfigError(
                f"pre-flight: estimated ${est:.2f} for "
                f"{self.estimate_calls(prompts_per_scenario)} calls exceeds budget "
                f"${self.budget.max_cost_usd:.2f}. Reduce scope, raise the budget, "
                "or set budget.preflight: false."
            )


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate a run_config.yaml file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    try:
        return RunConfig.model_validate(data)
    except Exception as e:  # noqa: BLE001 - surface any validation error as ConfigError
        raise ConfigError(f"invalid run config {path}: {e}") from e


def run_config_json_schema() -> dict[str, Any]:
    """JSON Schema for a run config, for editor validation or CI."""
    return RunConfig.model_json_schema()
