"""Evaluation pipeline for processing model outputs."""

import asyncio
from typing import Any
from uuid import uuid4

from fairbench.adapters.base import ModelAdapter
from fairbench.core.exceptions import EvaluationError
from fairbench.core.types import (
    EvaluatedOutput,
    EvaluationRun,
    GeneratedOutput,
    GenerationConfig,
    RunStatus,
)
from fairbench.counterfactual.generator import ExpandedPrompt
from fairbench.evaluation.base import Evaluator


class EvaluationPipeline:
    """Pipeline for generating and evaluating model outputs.

    The pipeline:
    1. Generates outputs from prompts using a model adapter
    2. Runs evaluators on each output (embeddings, toxicity, etc.)
    3. Produces EvaluatedOutput objects ready for metric computation
    """

    def __init__(
        self,
        model: ModelAdapter,
        evaluators: list[Evaluator] | None = None,
        concurrency: int = 10,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        """Initialize the evaluation pipeline.

        Args:
            model: Model adapter to use for generation.
            evaluators: List of evaluators to run on outputs.
            concurrency: Maximum concurrent operations.
            generation_config: Default generation configuration.
        """
        self.model = model
        self.evaluators = evaluators or []
        self.concurrency = concurrency
        self.generation_config = generation_config or GenerationConfig()
        self._semaphore = asyncio.Semaphore(concurrency)

    async def _generate_one(self, prompt: ExpandedPrompt) -> GeneratedOutput:
        """Generate output for a single prompt with rate limiting.

        Args:
            prompt: The expanded prompt to generate from.

        Returns:
            The generated output.
        """
        async with self._semaphore:
            return await self.model.generate(
                prompt.prompt,
                self.generation_config,
            )

    async def _evaluate_one(
        self, output: GeneratedOutput
    ) -> dict[str, Any]:
        """Run all evaluators on a single output.

        Args:
            output: The generated output.

        Returns:
            Combined evaluation results.
        """
        results: dict[str, Any] = {}

        for evaluator in self.evaluators:
            if evaluator.applies_to(output):
                try:
                    eval_result = await evaluator.evaluate(output)
                    results.update(eval_result)
                except Exception as e:
                    # Log but don't fail the whole pipeline
                    results[f"{evaluator.name}_error"] = str(e)

        return results

    async def process_prompt(
        self, prompt: ExpandedPrompt
    ) -> EvaluatedOutput:
        """Process a single prompt: generate and evaluate.

        Args:
            prompt: The expanded prompt.

        Returns:
            The evaluated output.
        """
        # Generate
        output = await self._generate_one(prompt)

        # Evaluate
        eval_results = await self._evaluate_one(output)

        # Combine into EvaluatedOutput
        return EvaluatedOutput(
            id=uuid4(),
            output=output,
            scenario_id=prompt.scenario_id,
            is_counterfactual=prompt.is_counterfactual,
            counterfactual_attribute=prompt.attribute,
            counterfactual_value=prompt.attribute_value,
            original_prompt=prompt.original_prompt,
            embedding=eval_results.get("embedding"),
            toxicity=eval_results.get("toxicity"),
            sentiment=eval_results.get("sentiment"),
            detected_entities=eval_results.get("entities", {}),
            custom_evaluations={
                k: v for k, v in eval_results.items()
                if k not in ("embedding", "toxicity", "sentiment", "entities")
            },
        )

    async def process_prompts(
        self, prompts: list[ExpandedPrompt]
    ) -> list[EvaluatedOutput]:
        """Process multiple prompts in parallel.

        Args:
            prompts: List of expanded prompts.

        Returns:
            List of evaluated outputs.
        """
        tasks = [self.process_prompt(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def run(
        self,
        prompts: list[ExpandedPrompt],
        run_id: str | None = None,
    ) -> list[EvaluatedOutput]:
        """Run the full evaluation pipeline.

        Args:
            prompts: List of expanded prompts to process.
            run_id: Optional run ID for tracking.

        Returns:
            List of evaluated outputs.

        Raises:
            EvaluationError: If the pipeline fails.
        """
        if not prompts:
            return []

        try:
            outputs = await self.process_prompts(prompts)
            return outputs
        except Exception as e:
            raise EvaluationError(f"Pipeline execution failed: {e}") from e

    async def run_batch_optimized(
        self, prompts: list[ExpandedPrompt]
    ) -> list[EvaluatedOutput]:
        """Run pipeline with batch-optimized evaluation.

        This version batches evaluations for efficiency when evaluators
        support batch processing.

        Args:
            prompts: List of expanded prompts.

        Returns:
            List of evaluated outputs.
        """
        if not prompts:
            return []

        # Step 1: Generate all outputs
        generation_tasks = [self._generate_one(prompt) for prompt in prompts]
        outputs = await asyncio.gather(*generation_tasks)

        # Step 2: Batch evaluate
        all_eval_results: list[dict[str, Any]] = [{} for _ in outputs]

        for evaluator in self.evaluators:
            # Filter outputs this evaluator applies to
            applicable_indices = [
                i for i, output in enumerate(outputs)
                if evaluator.applies_to(output)
            ]

            if not applicable_indices:
                continue

            applicable_outputs = [outputs[i] for i in applicable_indices]

            try:
                results = await evaluator.evaluate_batch(applicable_outputs)
                for idx, result in zip(applicable_indices, results):
                    all_eval_results[idx].update(result)
            except Exception as e:
                # Log error but continue
                for idx in applicable_indices:
                    all_eval_results[idx][f"{evaluator.name}_error"] = str(e)

        # Step 3: Combine into EvaluatedOutputs
        evaluated_outputs = []
        for prompt, output, eval_results in zip(prompts, outputs, all_eval_results):
            evaluated_outputs.append(
                EvaluatedOutput(
                    id=uuid4(),
                    output=output,
                    scenario_id=prompt.scenario_id,
                    is_counterfactual=prompt.is_counterfactual,
                    counterfactual_attribute=prompt.attribute,
                    counterfactual_value=prompt.attribute_value,
                    original_prompt=prompt.original_prompt,
                    embedding=eval_results.get("embedding"),
                    toxicity=eval_results.get("toxicity"),
                    sentiment=eval_results.get("sentiment"),
                    detected_entities=eval_results.get("entities", {}),
                    custom_evaluations={
                        k: v for k, v in eval_results.items()
                        if k not in ("embedding", "toxicity", "sentiment", "entities")
                    },
                )
            )

        return evaluated_outputs

    def add_evaluator(self, evaluator: Evaluator) -> None:
        """Add an evaluator to the pipeline.

        Args:
            evaluator: The evaluator to add.
        """
        self.evaluators.append(evaluator)

    def remove_evaluator(self, name: str) -> bool:
        """Remove an evaluator by name.

        Args:
            name: Name of the evaluator to remove.

        Returns:
            True if an evaluator was removed.
        """
        initial_len = len(self.evaluators)
        self.evaluators = [e for e in self.evaluators if e.name != name]
        return len(self.evaluators) < initial_len
