"""Soccer Image Fairness Benchmark — example usage.

This script runs the full image benchmarking pipeline using DALL-E 3
to generate soccer images and evaluates them for demographic fairness
across six metrics (RSI, ODE, CDS, HSI, SAR, DSI).

Requirements:
    pip install openai anthropic openai-clip Pillow httpx

Environment variables:
    OPENAI_API_KEY    — required for DALL-E 3
    ANTHROPIC_API_KEY — required for VisionAnalyzer (Claude Vision)

Run:
    python examples/soccer_image_benchmark.py
    python examples/soccer_image_benchmark.py --quick          # 1 scenario, no CLIP
    python examples/soccer_image_benchmark.py --save-images ./images
"""

import argparse
import asyncio
import json
from pathlib import Path

# Auto-load .env from the project root (or current directory)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; fall back to shell environment


async def run_benchmark(
    save_images: str | None = None,
    use_clip: bool = True,
    quick: bool = False,
) -> None:
    from fairbench.adapters.image.dalle import DALLEAdapter
    from fairbench.core.image_engine import ImageBenchEngine
    from fairbench.core.image_types import ImageGenerationConfig
    from fairbench.core.types import Distribution
    from fairbench.evaluation.image.clip_evaluator import CLIPEvaluator
    from fairbench.evaluation.image.vision_analyzer import VisionAnalyzer

    print("=" * 64)
    print("FAIRBench — Soccer Image Generation Fairness Benchmark")
    print("=" * 64)

    # ── Adapters ──────────────────────────────────────────────────────────────
    adapter = DALLEAdapter(
        model="gpt-image-1",
        save_dir=save_images,
        download_images=True,
    )

    vision_analyzer = VisionAnalyzer(
        model="claude-sonnet-4-6",
        fallback_on_error=True,
    )

    clip_evaluator = CLIPEvaluator(model_name="ViT-B/32") if use_clip else None

    # ── Engine ────────────────────────────────────────────────────────────────
    engine = ImageBenchEngine()

    # Load soccer scenarios from the built-in YAML
    scenarios_yaml = (
        Path(__file__).parent.parent
        / "src" / "fairbench" / "scenarios" / "image" / "soccer_player.yaml"
    )
    engine.scenario_registry.load_file(str(scenarios_yaml))

    # In quick mode, run only the base player-in-action scenario
    scenario_names = ["soccer_player_action"] if quick else ["soccer_player"]

    # Generation config: use "hd" for better quality in a real benchmark
    gen_config = ImageGenerationConfig(
        size="1024x1024",
        quality="auto",  # gpt-image-1: "low" | "medium" | "high" | "auto"
    )

    # ── Real-world demographic baseline for RSI/SAR ───────────────────────────
    # FIFA reports roughly 30% of registered players are female (2022).
    # Approximately 45% of top-flight players identify as non-white.
    # This baseline represents what "equitable" distribution looks like.
    real_world_baseline = Distribution(
        probabilities={
            "male": 0.70,
            "female": 0.30,
        }
    )

    print(f"\nScenarios : {scenario_names}")
    print(f"Quick mode: {quick}")
    print(f"CLIP      : {'enabled' if use_clip else 'disabled'}")
    print(f"Save dir  : {save_images or 'not saving'}")
    print()

    # ── Run ───────────────────────────────────────────────────────────────────
    run = await engine.evaluate(
        model=adapter,
        scenarios=scenario_names,
        vision_analyzer=vision_analyzer,
        clip_evaluator=clip_evaluator,
        baseline=real_world_baseline,
        generation_config=gen_config,
        concurrency=3,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print()
    print(f"Run ID        : {run.id}")
    print(f"Status        : {run.status.value}")
    print(f"Total images  : {run.total_images()}")
    print(f"Refused       : {run.refused_count()}")
    print()

    # Metric summary
    print("── Fairness Metrics ──────────────────────────────────────────")
    for mr in run.metric_results:
        value_str = f"{mr.value:.4f}" if mr.value == mr.value else "N/A"
        print(f"  {mr.metric_name:<6} {value_str}  ({mr.interpretation})")

    # Per-image analysis
    print()
    print("── Per-image Analysis (VisionAnalyzer) ──────────────────────")
    print(f"  {'Scenario':<35} {'CF Attr':<12} {'CF Value':<16} {'Gender':<12} {'Skin':<8} {'Setting':<22} {'Quality'}")
    print("  " + "-" * 120)
    for ei in run.evaluated_images:
        va = ei.vision_analysis
        if va is None:
            continue
        cf_attr = ei.counterfactual_attribute or "-"
        cf_val = ei.counterfactual_value or "(base)"
        gender = va.perceived_gender[:10]
        skin = str(va.skin_tone_fitzpatrick or "-")
        setting = va.setting[:20]
        quality = f"{va.image_quality_score:.1f}" if va.image_quality_score else "-"
        print(f"  {ei.scenario_id:<35} {cf_attr:<12} {cf_val:<16} {gender:<12} {skin:<8} {setting:<22} {quality}")

    # CLIP probe comparison (if enabled)
    if use_clip:
        print()
        print("── CLIP Gender Probe Similarities (base images only) ────────")
        base_images = [ei for ei in run.evaluated_images if not ei.is_counterfactual]
        if base_images:
            print(f"  {'Scenario':<35} {'male_sim':<12} {'female_sim':<12} {'inferred'}")
            print("  " + "-" * 70)
            for ei in base_images:
                sims = ei.clip_similarities
                if sims:
                    m = sims.get("gender_male", 0.0)
                    f = sims.get("gender_female", 0.0)
                    inferred = "male" if m > f + 0.01 else ("female" if f > m + 0.01 else "ambiguous")
                    print(f"  {ei.scenario_id:<35} {m:<12.4f} {f:<12.4f} {inferred}")

    # Gender distribution across all generated images
    gender_counts: dict[str, int] = {}
    for ei in run.evaluated_images:
        if ei.vision_analysis:
            g = ei.vision_analysis.perceived_gender
            gender_counts[g] = gender_counts.get(g, 0) + 1

    print()
    print("── Gender Distribution (all images) ─────────────────────────")
    total = sum(gender_counts.values())
    for g, count in sorted(gender_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total else 0
        print(f"  {g:<20} {count:3d}  ({pct:.1f}%)")

    # Stereotype summary
    all_stereotypes = []
    for ei in run.evaluated_images:
        if ei.vision_analysis and ei.vision_analysis.stereotypes_detected:
            all_stereotypes.extend(ei.vision_analysis.stereotypes_detected)

    if all_stereotypes:
        print()
        print("── Detected Stereotypes (from VisionAnalyzer) ──────────────")
        for s in all_stereotypes:
            print(f"  • {s}")

    # Save scorecard
    scorecard_path = Path("soccer_benchmark_scorecard.json")
    scorecard = engine.generate_scorecard(run)
    scorecard_path.write_text(json.dumps(scorecard, indent=2))
    print()
    print(f"Scorecard saved to: {scorecard_path}")

    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Soccer Image Fairness Benchmark")
    parser.add_argument(
        "--save-images", type=str, default=None,
        help="Directory to save generated images (default: don't save)"
    )
    parser.add_argument(
        "--no-clip", action="store_true",
        help="Disable CLIP evaluation (faster, no openai-clip dependency)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only 1 scenario (soccer_player_action) for quick testing"
    )
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            save_images=args.save_images,
            use_clip=not args.no_clip,
            quick=args.quick,
        )
    )


if __name__ == "__main__":
    main()
