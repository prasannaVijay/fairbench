"""JSON scorecard generation for FAIRBench evaluation runs."""

import statistics
from datetime import datetime, timezone
from typing import Any

from fairbench.core.types import Distribution, EvaluatedOutput, EvaluationRun
from fairbench.metrics.base import Metric
from fairbench.metrics.cds import CounterfactualDivergenceScore
from fairbench.metrics.hsi import HarmSeverityIndex
from fairbench.metrics.ode import OutputDiversityEntropy
from fairbench.metrics.rsi import RepresentationSkewIndex
from fairbench.metrics.sar import StereotypeAmplificationRatio

_DEFAULT_METRICS: dict[str, Metric] = {
    "CDS": CounterfactualDivergenceScore(),
    "RSI": RepresentationSkewIndex(),
    "SAR": StereotypeAmplificationRatio(),
    "ODE": OutputDiversityEntropy(),
    "HSI": HarmSeverityIndex(),
}


def generate_scorecard(
    run: EvaluationRun,
    baseline: Distribution | None = None,
    metrics: dict[str, Metric] | None = None,
) -> dict[str, Any]:
    """Generate a JSON-serializable scorecard for a completed evaluation run.

    The scorecard contains:
    - Run metadata (id, status, timing)
    - Model information
    - Aggregated metric summary (mean and median across scenarios)
    - Per-scenario metric breakdown

    Args:
        run: The completed evaluation run.
        baseline: Optional baseline distribution passed to metrics.
        metrics: Custom metric instances. Defaults to all five built-in metrics.

    Returns:
        A dict ready for json.dumps().
    """
    metric_registry = metrics if metrics is not None else _DEFAULT_METRICS

    # Group outputs by scenario_id
    by_scenario: dict[str, list[EvaluatedOutput]] = {}
    for output in run.outputs:
        by_scenario.setdefault(output.scenario_id, []).append(output)

    # Compute per-scenario metric scores
    scenario_details: dict[str, dict[str, Any]] = {}
    for scenario_id, outputs in by_scenario.items():
        metric_scores: dict[str, Any] = {}
        for metric_name in run.metrics_requested:
            metric = metric_registry.get(metric_name)
            if metric is None:
                continue
            try:
                result = metric.compute(outputs, baseline)
                metric_scores[metric_name] = {
                    "value": result.value,
                    "n_samples": result.n_samples,
                    "interpretation": result.interpretation,
                }
            except Exception as exc:
                metric_scores[metric_name] = {"error": str(exc)}

        base_count = sum(1 for o in outputs if not o.is_counterfactual)
        cf_count = sum(1 for o in outputs if o.is_counterfactual)
        scenario_details[scenario_id] = {
            "n_outputs": len(outputs),
            "n_base": base_count,
            "n_counterfactual": cf_count,
            "metrics": metric_scores,
        }

    # Aggregate mean/median across scenarios for each metric
    summary: dict[str, Any] = {}
    for metric_name in run.metrics_requested:
        values = [
            scenario_details[sid]["metrics"][metric_name]["value"]
            for sid in scenario_details
            if metric_name in scenario_details[sid]["metrics"]
            and "value" in scenario_details[sid]["metrics"][metric_name]
        ]

        if values:
            metric = metric_registry.get(metric_name)
            summary[metric_name] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "n_scenarios": len(values),
                "rating": _rate(statistics.mean(values), metric),
            }
        else:
            # Fall back to the run-level result if per-scenario couldn't be computed
            run_level = next(
                (mr for mr in run.metric_results if mr.metric_name == metric_name),
                None,
            )
            if run_level is not None:
                metric = metric_registry.get(metric_name)
                summary[metric_name] = {
                    "mean": run_level.value,
                    "median": run_level.value,
                    "n_scenarios": 0,
                    "rating": _rate(run_level.value, metric),
                }

    return {
        "fairbench_scorecard": True,
        "scorecard_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": {
            "id": str(run.id),
            "status": run.status.value,
            "scenario_sets": run.scenario_sets,
            "created_at": run.created_at.isoformat(),
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "error_message": run.error_message,
        },
        "model": run.model_info.model_dump(),
        "summary": summary,
        "details": {
            "by_scenario": scenario_details,
        },
    }


def _rate(value: float, metric: Metric | None) -> str:
    """Return 'good', 'acceptable', or 'poor' based on metric thresholds."""
    if metric is None:
        return "unknown"
    thresholds = metric.get_thresholds()
    if value <= thresholds.get("good", 0.1):
        return "good"
    elif value <= thresholds.get("acceptable", 0.3):
        return "acceptable"
    return "poor"
