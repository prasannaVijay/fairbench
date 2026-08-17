"""Soccer image fairness benchmark — offline stub run (no API key required).

This is the companion to Chapter 5. It runs the *same* ImageBenchEngine
pipeline as ``soccer_image_benchmark.py`` — scenario expansion, generation,
vision analysis, CLIP embeddings, six fairness metrics, and a scorecard — but
against the offline stub adapter and stub evaluators, so it produces a green
result in a couple of seconds with no network access and no keys.

Run:
    python examples/soccer_stub_benchmark.py

The recorded outputs are deterministic and intentionally skewed toward a
male default for unconstrained prompts, so the scorecard shows an explainable
representation signal. The numbers are illustrative, not measured.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running straight from a checkout without installing the package.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


async def main() -> int:
    from fairbench_genai.adapters.image.stub import StubImageAdapter
    from fairbench_genai.core.image_engine import ImageBenchEngine
    from fairbench_genai.core.image_types import ImageGenerationConfig
    from fairbench_genai.core.types import Distribution
    from fairbench_genai.evaluation.image.stub import StubCLIPEvaluator, StubVisionAnalyzer

    print("=" * 64)
    print("FAIRBench — Soccer Image Fairness Benchmark (offline stub)")
    print("=" * 64)

    engine = ImageBenchEngine()
    scenarios_yaml = (
        Path(__file__).resolve().parent.parent
        / "src" / "fairbench_genai" / "scenarios" / "image" / "soccer_player.yaml"
    )
    engine.scenario_registry.load_file(str(scenarios_yaml))

    # Baseline: an equitable gender split for the representation metrics.
    baseline = Distribution(probabilities={"male": 0.5, "female": 0.5})

    run = await engine.evaluate(
        model=StubImageAdapter(),
        scenarios=["soccer_player_action"],
        vision_analyzer=StubVisionAnalyzer(),
        clip_evaluator=StubCLIPEvaluator(),
        baseline=baseline,
        generation_config=ImageGenerationConfig(size="1024x1024", quality="standard"),
        concurrency=4,
    )

    print(f"\nRun ID       : {run.id}")
    print(f"Status       : {run.status.value}")
    print(f"Total images : {run.total_images()}")
    print(f"Refused      : {run.refused_count()}")

    print("\n-- Fairness metrics ------------------------------------------")
    for mr in run.metric_results:
        val = "nan" if mr.value != mr.value else f"{mr.value:.3f}"  # nan-safe
        print(f"  {mr.metric_name:>4} = {val:<7} {mr.interpretation or ''}")

    scorecard = engine.generate_scorecard(run)
    print("\n-- Scorecard: base scenario representation -------------------")
    for sid, s in scorecard.get("by_scenario", scorecard).items() if isinstance(
        scorecard.get("by_scenario", scorecard), dict
    ) else []:
        if not isinstance(s, dict) or "gender_distribution" not in s:
            continue
        print(f"  scenario         : {sid}")
        print(f"  images / refused : {s.get('n_images')} / {s.get('n_refused')}")
        print(f"  gender split     : {s.get('gender_distribution')}")
        print(f"  skin-tone split  : {s.get('skin_tone_distribution')}")

    # A run is "green" if it completed and produced at least one metric.
    ok = run.status.value == "completed" and len(run.metric_results) > 0
    print("\nResult:", "PASS (pipeline ran end to end)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
