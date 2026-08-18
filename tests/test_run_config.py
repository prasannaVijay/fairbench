"""Tests for the Chapter 5 hardened run configuration schema."""

from pathlib import Path

import pytest

from fairbench_genai.core.exceptions import ConfigError
from fairbench_genai.core.run_config import (
    RunConfig,
    RunOutcome,
    load_run_config,
    run_config_json_schema,
)

_EXAMPLE = Path(__file__).resolve().parent.parent / "ch05" / "run_config.yaml"


def test_example_run_config_loads() -> None:
    cfg = load_run_config(_EXAMPLE)
    assert cfg.run.id == "soccer_benchmark_run_20250301"
    assert cfg.model.capture_revised_prompt is True
    assert cfg.execution.retry.backoff_type == "exponential"
    assert cfg.execution.rate_limit.strategy == "token_bucket"
    assert cfg.budget.preflight is True
    assert cfg.provenance.schema_version == "1.0"


def _minimal(**budget: object) -> RunConfig:
    return RunConfig.model_validate(
        {
            "run": {"id": "r", "scenario_ids": ["s1", "s2"]},
            "model": {"provider": "openai", "model_id": "dall-e-3"},
            "sampling": {"replicates_per_prompt": 5},
            "budget": budget,
        }
    )


def test_preflight_rejects_over_budget_run() -> None:
    # 2 scenarios x 10 prompts x 5 replicates x $0.04 = $4.00 > $1.00 budget
    cfg = _minimal(max_cost_usd=1.0, cost_per_call_usd=0.04)
    with pytest.raises(ConfigError):
        cfg.preflight(prompts_per_scenario=10)


def test_preflight_allows_in_budget_run() -> None:
    cfg = _minimal(max_cost_usd=100.0, cost_per_call_usd=0.04)
    cfg.preflight(prompts_per_scenario=10)  # should not raise
    assert cfg.estimate_calls(10) == 100


def test_retry_statuses_must_be_disjoint() -> None:
    with pytest.raises(Exception):
        RunConfig.model_validate(
            {
                "run": {"id": "r", "scenario_ids": ["s1"]},
                "model": {"provider": "openai", "model_id": "m"},
                "execution": {"retry": {"retry_on_status": [429], "fail_fast_on_status": [429]}},
            }
        )


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(Exception):
        RunConfig.model_validate(
            {
                "run": {"id": "r", "scenario_ids": ["s1"]},
                "model": {"provider": "openai", "model_id": "m"},
                "budgett": {},  # typo
            }
        )


def test_outcome_enum_has_content_policy_refusal() -> None:
    assert RunOutcome.REFUSED_CONTENT_POLICY.value == "refused_content_policy"


def test_json_schema_exports() -> None:
    schema = run_config_json_schema()
    assert schema["title"] == "RunConfig"
