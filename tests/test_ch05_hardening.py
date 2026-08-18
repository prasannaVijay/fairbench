"""Tests for the Chapter 5 hardening: engine<->run_config wiring, adapter
cost/request-id metadata, and triage random-audit + queue cap."""

from pathlib import Path

import pytest

from fairbench_genai.adapters.image.stub import StubImageAdapter
from fairbench_genai.core.exceptions import ConfigError
from fairbench_genai.core.image_engine import ImageBenchEngine
from fairbench_genai.core.run_config import load_run_config
from fairbench_genai.core.types import Distribution, EvaluatedOutput, GeneratedOutput, GenerationConfig, ModelInfo, RunStatus
from fairbench_genai.evaluation.image.stub import StubCLIPEvaluator, StubVisionAnalyzer
from fairbench_genai.evaluation.triage import TriageRouter

_SOCCER = Path(__file__).resolve().parent.parent / "src" / "fairbench_genai" / "scenarios" / "image" / "soccer_player.yaml"
_RUN_CFG = Path(__file__).resolve().parent.parent / "ch05" / "run_config.yaml"


# ---- Engine <-> run_config wiring ------------------------------------------

@pytest.mark.asyncio
async def test_engine_runs_with_run_config() -> None:
    engine = ImageBenchEngine()
    engine.scenario_registry.load_file(str(_SOCCER))
    run = await engine.evaluate(
        model=StubImageAdapter(),
        scenarios=["soccer_player_action"],
        vision_analyzer=StubVisionAnalyzer(),
        clip_evaluator=StubCLIPEvaluator(),
        baseline=Distribution(probabilities={"male": 0.5, "female": 0.5}),
        run_config=load_run_config(_RUN_CFG),
    )
    assert run.status == RunStatus.COMPLETED
    assert run.total_images() > 0


@pytest.mark.asyncio
async def test_engine_preflight_rejects_over_budget() -> None:
    cfg = load_run_config(_RUN_CFG)
    # Force a tiny budget so the pre-flight estimate exceeds it.
    tiny = cfg.model_copy(update={"budget": cfg.budget.model_copy(update={"max_cost_usd": 0.01})})
    engine = ImageBenchEngine()
    engine.scenario_registry.load_file(str(_SOCCER))
    with pytest.raises(ConfigError):
        await engine.evaluate(
            model=StubImageAdapter(),
            scenarios=["soccer_player_action"],
            vision_analyzer=StubVisionAnalyzer(),
            clip_evaluator=StubCLIPEvaluator(),
            run_config=tiny,
        )


# ---- Adapter cost / request-id ---------------------------------------------

@pytest.mark.asyncio
async def test_adapter_returns_cost_and_request_id() -> None:
    img = await StubImageAdapter().generate("A professional soccer player in action during a match")
    assert img.metadata["cost_usd"] == 0.04
    assert img.metadata["request_id"].startswith("stub-")


# ---- Triage random audit + cap ---------------------------------------------

def _outputs(n: int, scenario: str = "s1") -> list[EvaluatedOutput]:
    go = GeneratedOutput(
        text="x", prompt="p",
        model_info=ModelInfo(name="m", provider="p"),
        generation_config=GenerationConfig(),
    )
    return [EvaluatedOutput(output=go, scenario_id=scenario) for _ in range(n)]


def test_random_audit_samples_unflagged_outputs() -> None:
    router = TriageRouter(min_human_review_rate=0.0, random_audit_rate=0.1)
    flags = router.triage(_outputs(50))
    audit = [f for f in flags if f.layer == "audit"]
    assert len(audit) == 5  # round(0.1 * 50)


def test_queue_cap_limits_backlog() -> None:
    router = TriageRouter(min_human_review_rate=0.0, random_audit_rate=0.5, max_review_queue=3)
    flags = router.triage(_outputs(50))
    assert len(flags) <= 3
