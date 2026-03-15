"""Main FAIRBench engine that orchestrates evaluations."""

import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fairbench.adapters.base import ModelAdapter
from fairbench.adapters.registry import get_adapter_registry
from fairbench.core.config import Config, get_config
from fairbench.core.exceptions import FairBenchError
from fairbench.core.types import (
    Distribution,
    EvaluatedOutput,
    EvaluationRun,
    GenerationConfig,
    MetricResult,
    ModelInfo,
    RunStatus,
    Scenario,
)
from fairbench.counterfactual.generator import CounterfactualGenerator, ExpandedPrompt
from fairbench.evaluation.embeddings import EmbeddingEvaluator
from fairbench.evaluation.pipeline import EvaluationPipeline
from fairbench.evaluation.sentiment import SentimentEvaluator
from fairbench.evaluation.toxicity import ToxicityEvaluator
from fairbench.metrics.base import Metric
from fairbench.metrics.cds import CounterfactualDivergenceScore
from fairbench.metrics.hsi import HarmSeverityIndex
from fairbench.metrics.ode import OutputDiversityEntropy
from fairbench.metrics.rsi import RepresentationSkewIndex
from fairbench.metrics.sar import StereotypeAmplificationRatio
from fairbench.scenarios.registry import ScenarioRegistry, get_registry
from fairbench.storage.base import StorageBackend
from fairbench.storage.sqlite import SQLiteBackend


class FairBenchEngine:
    """Main engine for running fairness evaluations.

    The engine coordinates:
    - Scenario loading and expansion
    - Model generation via adapters
    - Output evaluation (embeddings, toxicity, etc.)
    - Metric computation
    - Result storage and reporting
    """

    # Default metrics
    DEFAULT_METRICS = ["CDS", "RSI", "SAR", "ODE", "HSI"]

    def __init__(
        self,
        config: Config | None = None,
        storage: StorageBackend | None = None,
    ) -> None:
        """Initialize the FAIRBench engine.

        Args:
            config: Configuration. If None, loads from default locations.
            storage: Storage backend. If None, uses SQLite.
        """
        self.config = config or get_config()
        self.storage = storage or SQLiteBackend(self.config.storage.sqlite_path)
        self.scenario_registry = get_registry()
        self.adapter_registry = get_adapter_registry()
        self.counterfactual_generator = CounterfactualGenerator()

        # Initialize metrics
        self._metrics: dict[str, Metric] = {
            "CDS": CounterfactualDivergenceScore(),
            "RSI": RepresentationSkewIndex(),
            "SAR": StereotypeAmplificationRatio(),
            "ODE": OutputDiversityEntropy(),
            "HSI": HarmSeverityIndex(),
        }

    def register_adapter(self, name: str, adapter: ModelAdapter) -> None:
        """Register a custom model adapter.

        Args:
            name: Name to register the adapter under.
            adapter: The adapter instance.
        """
        self.adapter_registry.register(name, adapter)

    def register_metric(self, metric: Metric) -> None:
        """Register a custom metric.

        Args:
            metric: The metric to register.
        """
        self._metrics[metric.name] = metric

    async def evaluate(
        self,
        model: str | ModelAdapter,
        scenarios: list[str] | list[Scenario],
        metrics: list[str] | None = None,
        baseline: Distribution | None = None,
        generation_config: GenerationConfig | None = None,
        concurrency: int = 10,
        save_run: bool = True,
    ) -> EvaluationRun:
        """Run a fairness evaluation.

        Args:
            model: Model adapter name or instance.
            scenarios: Scenario set names or Scenario objects.
            metrics: Metric names to compute. If None, uses all defaults.
            baseline: Baseline distribution for metrics.
            generation_config: Configuration for text generation.
            concurrency: Maximum concurrent API calls.
            save_run: Whether to persist the run to storage.

        Returns:
            The completed evaluation run.
        """
        # Resolve model adapter
        if isinstance(model, str):
            adapter = self.adapter_registry.get(model)
        else:
            adapter = model

        # Resolve scenarios
        scenario_objs = self._resolve_scenarios(scenarios)

        # Create run record
        run = EvaluationRun(
            id=uuid4(),
            status=RunStatus.PENDING,
            model_info=adapter.get_model_info(),
            scenario_sets=[s if isinstance(s, str) else "custom" for s in scenarios],
            metrics_requested=metrics or self.DEFAULT_METRICS,
            config_snapshot={
                "generation_config": (generation_config or GenerationConfig()).model_dump(),
                "concurrency": concurrency,
            },
        )

        if save_run:
            await self.storage.save_run(run)

        try:
            # Update status
            run = EvaluationRun(
                **{**run.model_dump(), "status": RunStatus.RUNNING, "started_at": datetime.now(timezone.utc)}
            )
            if save_run:
                await self.storage.update_run(run)

            # Expand scenarios into prompts
            prompts = self._expand_scenarios(scenario_objs)

            # Create evaluation pipeline
            pipeline = self._create_pipeline(
                adapter, generation_config, concurrency
            )

            # Run evaluation
            outputs = await pipeline.run_batch_optimized(prompts)

            # Compute metrics
            metric_results = self._compute_metrics(
                outputs, metrics or self.DEFAULT_METRICS, baseline
            )

            # Update run with results
            run = EvaluationRun(
                **{
                    **run.model_dump(),
                    "status": RunStatus.COMPLETED,
                    "completed_at": datetime.now(timezone.utc),
                    "outputs": outputs,
                    "metric_results": metric_results,
                }
            )

        except Exception as e:
            run = EvaluationRun(
                **{
                    **run.model_dump(),
                    "status": RunStatus.FAILED,
                    "completed_at": datetime.now(timezone.utc),
                    "error_message": str(e),
                }
            )
            if save_run:
                await self.storage.update_run(run)
            raise FairBenchError(f"Evaluation failed: {e}") from e

        if save_run:
            await self.storage.update_run(run)

        return run

    def _resolve_scenarios(
        self, scenarios: list[str] | list[Scenario]
    ) -> list[Scenario]:
        """Resolve scenario references to Scenario objects."""
        result = []
        for s in scenarios:
            if isinstance(s, Scenario):
                result.append(s)
            elif isinstance(s, str):
                # Try as scenario set name first
                try:
                    scenario_set = self.scenario_registry.get_set(s)
                    result.extend(scenario_set.scenarios)
                except Exception:
                    # Try as individual scenario ID
                    scenario = self.scenario_registry.get_scenario(s)
                    result.append(scenario)
        return result

    def _expand_scenarios(self, scenarios: list[Scenario]) -> list[ExpandedPrompt]:
        """Expand scenarios into prompts including counterfactuals."""
        return self.counterfactual_generator.expand_scenarios(scenarios)

    def _create_pipeline(
        self,
        adapter: ModelAdapter,
        generation_config: GenerationConfig | None,
        concurrency: int,
    ) -> EvaluationPipeline:
        """Create the evaluation pipeline with evaluators."""
        evaluators = [
            EmbeddingEvaluator(device="cpu"),
            ToxicityEvaluator(backend="local"),
            SentimentEvaluator(),
        ]

        return EvaluationPipeline(
            model=adapter,
            evaluators=evaluators,
            concurrency=concurrency,
            generation_config=generation_config,
        )

    def _compute_metrics(
        self,
        outputs: list[EvaluatedOutput],
        metric_names: list[str],
        baseline: Distribution | None,
    ) -> list[MetricResult]:
        """Compute requested metrics from evaluated outputs."""
        results = []

        for name in metric_names:
            if name not in self._metrics:
                continue

            metric = self._metrics[name]
            try:
                result = metric.compute(outputs, baseline)
                results.append(result)
            except Exception as e:
                # Record error but continue with other metrics
                results.append(
                    MetricResult(
                        metric_name=name,
                        value=float("nan"),
                        n_samples=len(outputs),
                        interpretation=f"Error: {e}",
                    )
                )

        return results

    async def get_run(self, run_id: str) -> EvaluationRun | None:
        """Get an evaluation run by ID.

        Args:
            run_id: The run ID.

        Returns:
            The evaluation run, or None if not found.
        """
        return await self.storage.get_run(run_id)

    async def list_runs(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Any]:
        """List recent evaluation runs.

        Args:
            limit: Maximum number of runs to return.
            offset: Number of runs to skip.

        Returns:
            List of run summaries.
        """
        return await self.storage.list_runs(limit=limit, offset=offset)

    def get_available_metrics(self) -> dict[str, str]:
        """Get available metrics and their descriptions.

        Returns:
            Dictionary mapping metric names to descriptions.
        """
        return {name: metric.description for name, metric in self._metrics.items()}

    def get_available_scenarios(self) -> list[str]:
        """Get available scenario set names.

        Returns:
            List of registered scenario set names.
        """
        return self.scenario_registry.list_sets()

    async def close(self) -> None:
        """Close the engine and release resources."""
        await self.storage.close()
