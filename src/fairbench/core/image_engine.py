"""ImageBenchEngine: orchestrates image generation fairness benchmarking.

Mirrors FairBenchEngine but operates on GeneratedImage / EvaluatedImage
instead of text. After analysis, bridges to EvaluatedOutput so the
existing six fairness metrics can be reused unchanged.

Typical usage:
    engine = ImageBenchEngine()
    run = await engine.evaluate(
        model=DALLEAdapter(),
        scenarios=["soccer_player"],
        vision_analyzer=VisionAnalyzer(),
        clip_evaluator=CLIPEvaluator(),
    )
"""

import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fairbench.adapters.image.base import ImageModelAdapter
from fairbench.core.exceptions import FairBenchError
from fairbench.core.image_types import (
    EvaluatedImage,
    ImageAnalysis,
    ImageEvaluationRun,
    ImageGenerationConfig,
)
from fairbench.core.types import (
    Distribution,
    EvaluatedOutput,
    MetricResult,
    RunStatus,
    Scenario,
)
from fairbench.counterfactual.generator import CounterfactualGenerator, ExpandedPrompt
from fairbench.evaluation.image.clip_evaluator import CLIPEvaluator
from fairbench.evaluation.image.vision_analyzer import VisionAnalyzer
from fairbench.metrics.base import Metric
from fairbench.metrics.cds import CounterfactualDivergenceScore
from fairbench.metrics.dsi import DifferentialServiceIndex
from fairbench.metrics.hsi import HarmSeverityIndex
from fairbench.metrics.ode import OutputDiversityEntropy
from fairbench.metrics.rsi import RepresentationSkewIndex
from fairbench.metrics.sar import StereotypeAmplificationRatio
from fairbench.scenarios.registry import ScenarioRegistry, get_registry


class ImageBenchEngine:
    """Engine for image generation fairness evaluations.

    Orchestrates:
      1. Scenario expansion (same YAML format, same counterfactual generator)
      2. Image generation via ImageModelAdapter
      3. Analysis via VisionAnalyzer (Claude Vision) + CLIPEvaluator
      4. Bridge: EvaluatedImage → EvaluatedOutput
      5. Metric computation via the existing six fairness metrics
      6. Scorecard generation
    """

    DEFAULT_METRICS = ["RSI", "ODE", "CDS", "HSI", "SAR", "DSI"]

    def __init__(self) -> None:
        self.scenario_registry: ScenarioRegistry = get_registry()
        self.counterfactual_generator = CounterfactualGenerator()

        self._metrics: dict[str, Metric] = {
            "RSI": RepresentationSkewIndex(),
            "ODE": OutputDiversityEntropy(),
            "CDS": CounterfactualDivergenceScore(),
            "HSI": HarmSeverityIndex(),
            "SAR": StereotypeAmplificationRatio(),
            "DSI": DifferentialServiceIndex(),
        }

    def register_metric(self, metric: Metric) -> None:
        self._metrics[metric.name] = metric

    async def evaluate(
        self,
        model: ImageModelAdapter,
        scenarios: list[str] | list[Scenario],
        vision_analyzer: VisionAnalyzer | None = None,
        clip_evaluator: CLIPEvaluator | None = None,
        metrics: list[str] | None = None,
        baseline: Distribution | None = None,
        generation_config: ImageGenerationConfig | None = None,
        concurrency: int = 5,
    ) -> ImageEvaluationRun:
        """Run an image fairness evaluation.

        Args:
            model: Image generation adapter (DALLEAdapter, StableDiffusionAdapter, …).
            scenarios: Scenario set names or Scenario objects.
            vision_analyzer: Claude Vision analyzer. Instantiated with defaults if None.
            clip_evaluator: CLIP evaluator. Instantiated with defaults if None.
            metrics: Metric names to compute. Defaults to all six.
            baseline: Baseline distribution (e.g., real-world demographic data).
            generation_config: Image generation parameters.
            concurrency: Max concurrent image generation API calls.

        Returns:
            Completed ImageEvaluationRun with metrics and per-image analysis.
        """
        vision_analyzer = vision_analyzer or VisionAnalyzer()
        clip_evaluator = clip_evaluator or CLIPEvaluator()
        metrics = metrics or self.DEFAULT_METRICS
        generation_config = generation_config or ImageGenerationConfig()

        scenario_objs = self._resolve_scenarios(scenarios)
        prompts = self.counterfactual_generator.expand_scenarios(scenario_objs)

        run = ImageEvaluationRun(
            id=uuid4(),
            status=RunStatus.PENDING,
            model_info=model.get_model_info(),
            scenario_sets=[s if isinstance(s, str) else "custom" for s in scenarios],
            metrics_requested=metrics,
            config_snapshot={
                "generation_config": generation_config.model_dump(),
                "concurrency": concurrency,
                "vision_model": vision_analyzer.model,
                "clip_model": clip_evaluator.model_name,
            },
        )

        try:
            run = ImageEvaluationRun(
                **{**run.model_dump(), "status": RunStatus.RUNNING, "started_at": datetime.now(timezone.utc)}
            )

            # Step 1: Generate images
            print(f"  Generating {len(prompts)} images (concurrency={concurrency})…")
            from fairbench.core.image_types import GeneratedImage
            raw_images = await self._generate_images(
                model, prompts, generation_config, concurrency
            )

            # Step 2: Analyze images
            print("  Running VisionAnalyzer (Claude Vision)…")
            vision_analyses = await vision_analyzer.analyze_batch(
                [img for img in raw_images]
            )

            print("  Running CLIPEvaluator…")
            clip_results = await clip_evaluator.analyze_batch(raw_images)

            # Step 3: Assemble EvaluatedImage objects
            evaluated_images = self._assemble_evaluated_images(
                prompts, raw_images, vision_analyses, clip_results
            )

            # Step 4: Bridge to EvaluatedOutput for metric computation
            evaluated_outputs = [ei.to_evaluated_output() for ei in evaluated_images]

            # Step 5: Compute metrics
            print("  Computing fairness metrics…")
            metric_results = self._compute_metrics(evaluated_outputs, metrics, baseline)

            run = ImageEvaluationRun(
                **{
                    **run.model_dump(),
                    "status": RunStatus.COMPLETED,
                    "completed_at": datetime.now(timezone.utc),
                    "evaluated_images": evaluated_images,
                    "metric_results": metric_results,
                }
            )

        except Exception as e:
            run = ImageEvaluationRun(
                **{
                    **run.model_dump(),
                    "status": RunStatus.FAILED,
                    "completed_at": datetime.now(timezone.utc),
                    "error_message": str(e),
                }
            )
            raise FairBenchError(f"Image evaluation failed: {e}") from e

        return run

    async def _generate_images(
        self,
        model: ImageModelAdapter,
        prompts: list[ExpandedPrompt],
        config: ImageGenerationConfig,
        concurrency: int,
    ):
        """Generate images with concurrency limiting."""
        semaphore = asyncio.Semaphore(concurrency)

        async def _one(prompt: ExpandedPrompt):
            async with semaphore:
                return await model.generate(prompt.prompt, config)

        return list(await asyncio.gather(*[_one(p) for p in prompts]))

    def _assemble_evaluated_images(
        self,
        prompts: list[ExpandedPrompt],
        raw_images: list[Any],
        vision_analyses: list[ImageAnalysis],
        clip_results: list[tuple[list[float] | None, dict[str, float]]],
    ) -> list[EvaluatedImage]:
        """Combine generation + analysis results into EvaluatedImage objects."""
        evaluated = []
        for prompt, image, analysis, (clip_emb, clip_sims) in zip(
            prompts, raw_images, vision_analyses, clip_results
        ):
            # Check if image was refused (no content available)
            is_refused = not image.has_image() or image.metadata.get("refused", False)

            evaluated.append(
                EvaluatedImage(
                    id=uuid4(),
                    image=image,
                    scenario_id=prompt.scenario_id,
                    is_counterfactual=prompt.is_counterfactual,
                    counterfactual_attribute=prompt.attribute,
                    counterfactual_value=prompt.attribute_value,
                    original_prompt=prompt.original_prompt,
                    clip_embedding=clip_emb,
                    clip_similarities=clip_sims,
                    vision_analysis=analysis,
                    is_refused=is_refused,
                    refusal_reason=image.metadata.get("refusal_reason") if is_refused else None,
                )
            )
        return evaluated

    def _compute_metrics(
        self,
        outputs: list[EvaluatedOutput],
        metric_names: list[str],
        baseline: Distribution | None,
    ) -> list[MetricResult]:
        """Run requested metrics over the bridged outputs."""
        results = []
        for name in metric_names:
            if name not in self._metrics:
                continue
            metric = self._metrics[name]
            try:
                result = metric.compute(outputs, baseline)
                results.append(result)
            except Exception as e:
                results.append(
                    MetricResult(
                        metric_name=name,
                        value=float("nan"),
                        n_samples=len(outputs),
                        interpretation=f"Error: {e}",
                    )
                )
        return results

    def _resolve_scenarios(
        self, scenarios: list[str] | list[Scenario]
    ) -> list[Scenario]:
        result = []
        for s in scenarios:
            if isinstance(s, Scenario):
                result.append(s)
            elif isinstance(s, str):
                try:
                    scenario_set = self.scenario_registry.get_set(s)
                    result.extend(scenario_set.scenarios)
                except Exception:
                    result.append(self.scenario_registry.get_scenario(s))
        return result

    def generate_scorecard(self, run: ImageEvaluationRun) -> dict[str, Any]:
        """Generate a JSON-serializable scorecard from a completed run."""
        if run.status != RunStatus.COMPLETED:
            return {
                "run_id": str(run.id),
                "status": run.status.value,
                "error": run.error_message,
            }

        # Per-scenario breakdown
        by_scenario: dict[str, Any] = {}
        for ei in run.evaluated_images:
            sid = ei.scenario_id
            if sid not in by_scenario:
                by_scenario[sid] = {
                    "n_images": 0,
                    "n_base": 0,
                    "n_counterfactual": 0,
                    "n_refused": 0,
                    "gender_distribution": {},
                    "skin_tone_distribution": {},
                    "setting_distribution": {},
                    "avg_quality": [],
                    "stereotypes": [],
                }
            s = by_scenario[sid]
            s["n_images"] += 1
            if ei.is_refused:
                s["n_refused"] += 1
            if ei.is_counterfactual:
                s["n_counterfactual"] += 1
            else:
                s["n_base"] += 1
            if ei.vision_analysis:
                va = ei.vision_analysis
                gender = va.perceived_gender
                s["gender_distribution"][gender] = s["gender_distribution"].get(gender, 0) + 1
                tone = va.skin_tone_label
                s["skin_tone_distribution"][tone] = s["skin_tone_distribution"].get(tone, 0) + 1
                setting = va.setting
                s["setting_distribution"][setting] = s["setting_distribution"].get(setting, 0) + 1
                if va.image_quality_score is not None:
                    s["avg_quality"].append(va.image_quality_score)
                if va.stereotypes_detected:
                    s["stereotypes"].extend(va.stereotypes_detected)

        # Convert avg_quality lists to means
        for s in by_scenario.values():
            q = s.pop("avg_quality")
            s["avg_quality"] = sum(q) / len(q) if q else None

        # Metric summary
        metric_summary = {}
        for mr in run.metric_results:
            metric_summary[mr.metric_name] = {
                "value": mr.value,
                "n_samples": mr.n_samples,
                "interpretation": mr.interpretation,
                "details": mr.details,
            }

        elapsed = None
        if run.started_at and run.completed_at:
            elapsed = (run.completed_at - run.started_at).total_seconds()

        return {
            "run": {
                "id": str(run.id),
                "status": run.status.value,
                "modality": "image",
                "scenario_sets": run.scenario_sets,
                "created_at": run.created_at.isoformat(),
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "elapsed_seconds": elapsed,
            },
            "model": run.model_info.model_dump(),
            "summary": {
                "total_images": run.total_images(),
                "refused_count": run.refused_count(),
                "refused_rate": run.refused_count() / run.total_images() if run.total_images() else 0,
            },
            "metrics": metric_summary,
            "by_scenario": by_scenario,
        }
