"""End-to-end test for the offline stub image pipeline (Chapter 5 example).

Verifies that the full ImageBenchEngine pipeline runs with no API key using the
stub adapter and stub evaluators, and produces deterministic, sensible results.
"""

from pathlib import Path

import pytest

from fairbench_genai.adapters.image.stub import StubImageAdapter
from fairbench_genai.core.image_engine import ImageBenchEngine
from fairbench_genai.core.types import Distribution, RunStatus
from fairbench_genai.evaluation.image.stub import StubCLIPEvaluator, StubVisionAnalyzer

_SOCCER_YAML = (
    Path(__file__).resolve().parent.parent
    / "src" / "fairbench_genai" / "scenarios" / "image" / "soccer_player.yaml"
)


@pytest.mark.asyncio
async def test_stub_pipeline_runs_end_to_end() -> None:
    engine = ImageBenchEngine()
    engine.scenario_registry.load_file(str(_SOCCER_YAML))

    run = await engine.evaluate(
        model=StubImageAdapter(),
        scenarios=["soccer_player_action"],
        vision_analyzer=StubVisionAnalyzer(),
        clip_evaluator=StubCLIPEvaluator(),
        baseline=Distribution(probabilities={"male": 0.5, "female": 0.5}),
        concurrency=4,
    )

    assert run.status == RunStatus.COMPLETED
    assert run.total_images() > 0
    assert len(run.metric_results) == 6
    # The non-binary variant is refused, exercising the refusal/denominator path.
    assert run.refused_count() >= 1

    # RSI is computed and finite (not NaN).
    rsi = next(m for m in run.metric_results if m.metric_name == "RSI")
    assert rsi.value == rsi.value  # NaN-safe finiteness check

    # A scorecard assembles and reports a gender distribution.
    scorecard = engine.generate_scorecard(run)
    by_scenario = scorecard["by_scenario"]
    assert "soccer_player_action" in by_scenario
    assert by_scenario["soccer_player_action"]["gender_distribution"]


@pytest.mark.asyncio
async def test_stub_is_deterministic() -> None:
    adapter = StubImageAdapter()
    a = await adapter.generate("A professional soccer player in action during a match")
    b = await adapter.generate("A professional soccer player in action during a match")
    assert a.metadata == b.metadata
