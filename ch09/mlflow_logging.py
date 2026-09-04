"""Log a fairness benchmark result to an MLflow run.

Book: Chapter 9, "Linking results to model artifacts".

The pattern the chapter argues for is co-location: when a benchmark run
completes, the result is attached to the model version it evaluated instead of
being filed away in a separate logging system. MLflow gives three places to put
it, and this module uses all three. Tags carry the identity of the run (model
version, scenario, trigger type, gate decision), metrics carry the nine values
the gate policy judges, and an artifact carries the full result document so
that nothing in the record depends on the metric names staying stable.

Two corrections to the snippet as it appears in print
-----------------------------------------------------
The typeset listing does not run, and since the book cannot be changed, the
corrections live here:

1. The ``ODE_skin_tone`` and ``DSI`` lines are indented one level deeper than
   the lines around them, which raises ``IndentationError`` before anything
   executes. All nine ``log_metric`` calls belong at the same level inside the
   ``with`` block.
2. ``mlflow.log_artifact('result_{run_name}.yaml')`` is missing its ``f``
   prefix, so it looks for a file named literally ``result_{run_name}.yaml``
   rather than the run's own result file. It is an f-string here, and the path
   is passed in rather than reconstructed, which removes the chance of the two
   drifting apart.

MLflow is an optional import. The module is importable, and its result document
can be built and inspected, with mlflow absent; only :func:`log_benchmark_run`
needs the package, and it says so plainly when it is missing. That keeps the
test suite and an offline CI run free of a dependency this chapter uses for
illustration.

    pip install mlflow
    python ch09/mlflow_logging.py ch09/examples/result_soft_breach.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

try:  # pragma: no cover - exercised only where mlflow is installed
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore[assignment]

# The nine metric columns the Chapter 9 schema stores and the gate policy
# judges. Logging them by an explicit list keeps the MLflow record and the
# fairness_results table in step: a metric added to one has to be added here.
METRIC_NAMES = (
    "RSI_gender",
    "RSI_skin_tone",
    "ODE_gender",
    "ODE_skin_tone",
    "CDS_gender",
    "CDS_skin_tone",
    "SAR_gender",
    "HSI",
    "DSI",
)


def mlflow_available() -> bool:
    """Whether the optional mlflow dependency is importable."""
    return mlflow is not None


def build_result_document(
    run_id: str,
    model_version: str,
    scenario_id: str,
    trigger_type: str,
    metrics: dict[str, float | None],
    gate_decision: dict[str, Any] | str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the full result artifact that gets logged alongside the metrics.

    The individual metric values are logged as MLflow metrics so they can be
    charted and compared across runs. This document is logged as the artifact,
    and it is the record an auditor reads: it carries the values together with
    the model version, the scenario, the trigger that caused the run, and the
    evaluation context that makes the numbers reconstructable.
    """
    return {
        "run_id": run_id,
        "model_version": model_version,
        "scenario_id": scenario_id,
        "trigger_type": trigger_type,
        "metrics": {name: metrics.get(name) for name in METRIC_NAMES},
        "gate_decision": gate_decision,
        "context": context or {},
    }


def write_result_artifact(document: dict[str, Any], directory: str | Path = ".") -> Path:
    """Write the result document to ``result_<run_id>.yaml`` and return the path.

    The chapter's snippet names the artifact after the run. Building the path
    once, here, and handing it to :func:`log_benchmark_run` is what keeps the
    filename that gets written and the filename that gets logged identical.
    """
    run_name = document["run_id"]
    path = Path(directory) / f"result_{run_name}.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def log_benchmark_run(
    model_version: str,
    scenario_id: str,
    trigger_type: str,
    metrics: dict[str, float | None],
    gate_decision: dict[str, Any] | str,
    artifact_path: str | Path | None = None,
    tracking_uri: str | None = None,
) -> None:
    """Log one benchmark run to MLflow: tags, the nine metrics, and the artifact.

    Args:
        model_version: The registry URI or provider version string the run
            evaluated. This is the field that turns a collection of metric
            values into an audit trail, so it is required.
        scenario_id: The scenario the run covered, keyed to the Chapter 6
            scenario definition.
        trigger_type: What caused the run, from ch09/triggers.yaml.
        metrics: Metric name to value. A metric absent from this mapping is not
            logged, rather than being logged as zero.
        gate_decision: The decision from ch09/gate_evaluator.py.
        artifact_path: The result document written by
            :func:`write_result_artifact`. Skipped when None.
        tracking_uri: An MLflow tracking server. Defaults to MLflow's own
            resolution, which is a local ``mlruns/`` directory.

    Raises:
        RuntimeError: If mlflow is not installed.
    """
    if mlflow is None:
        raise RuntimeError(
            "mlflow is not installed. This module is illustrative for Chapter 9; "
            "install it with 'pip install mlflow' to log a run."
        )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    run_name = f"fairness_benchmark_{model_version}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_version", model_version)
        mlflow.set_tag("scenario", scenario_id)
        mlflow.set_tag("trigger_type", trigger_type)

        # Log individual metrics. All nine calls sit at one indentation level;
        # in the printed listing ODE_skin_tone and DSI are indented deeper,
        # which is an IndentationError.
        for name in METRIC_NAMES:
            value = metrics.get(name)
            if value is not None:
                mlflow.log_metric(name, float(value))

        # Log full result artifact. The printed listing writes
        # 'result_{run_name}.yaml' without the f prefix, which logs a file with
        # a literal brace in its name.
        if artifact_path is not None:
            mlflow.log_artifact(str(artifact_path))

        # Log gate decision
        decision = (
            gate_decision if isinstance(gate_decision, str) else gate_decision.get("decision")
        )
        mlflow.set_tag("gate_decision", decision)


def main(argv: list[str]) -> int:
    """Build (and, where mlflow is installed, log) a result from a result file."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gate_evaluator import evaluate_result, load_metrics  # noqa: PLC0415

    if not argv:
        print("usage: python ch09/mlflow_logging.py <result.yaml> [--log]")
        return 2

    path = Path(argv[0])
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    metrics = load_metrics(path)
    report = evaluate_result(metrics)
    document = build_result_document(
        run_id=raw.get("run_id", path.stem),
        model_version=raw.get("model_version", "unknown"),
        scenario_id=raw.get("scenario_id", "unknown"),
        trigger_type=raw.get("trigger_type", "manual"),
        metrics=metrics,
        gate_decision=report.to_dict(),
    )
    print(yaml.safe_dump(document, sort_keys=False))

    if "--log" in argv:
        if not mlflow_available():
            print("mlflow is not installed; nothing was logged.")
            return 1
        artifact = write_result_artifact(document)
        log_benchmark_run(
            model_version=document["model_version"],
            scenario_id=document["scenario_id"],
            trigger_type=document["trigger_type"],
            metrics=metrics,
            gate_decision=document["gate_decision"],
            artifact_path=artifact,
        )
        print(f"logged to MLflow with artifact {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
