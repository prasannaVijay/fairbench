"""``python -m fairbench.run``: the benchmark step of the Chapter 9 CI workflow.

Book: Chapter 9, "Wiring triggers to CI".

The workflow printed in the chapter ends with one command::

    python -m fairbench.run \
      --trigger ${{ inputs.trigger_type }} \
      --scenario ${{ inputs.scenario_id }} \
      --model-version ${{ inputs.model_version }}

This module is that command. It validates the three arguments, runs the
benchmark through ``fairbench_genai``, evaluates the result against
``ch09/gates.yaml`` with any live exception from ``ch09/exception_log.yaml``
applied, prints the decision, and exits non-zero when a hard gate blocks the
deployment. It computes no metric of its own; the measurement all happens in
the library, and what this module adds is the deployment layer around it.

Why ``--model-version`` is required
-----------------------------------
The chapter is unambiguous: tying every run to the exact model artifact being
evaluated is what makes a set of metric values into an audit trail. A run with
no version attached is a number nobody can trace back afterwards, so the
argument is required and a missing one is a usage error rather than a default.

The value takes a model-registry URI (``models:/soccer-gen/1.3``,
``registry://models/soccer-gen@1.3``) or a provider version string
(``gpt-image-1-2026-03-01``). A bare 40-character commit SHA is rejected: the
chapter reserves ``github.sha`` for pipeline-code provenance, and a commit of
the benchmark repository does not identify the model that was evaluated.

Running it
----------
    python -m fairbench.run --help
    python -m fairbench.run --trigger full_benchmark \
        --scenario soccer_pilot_v1 --model-version models:/soccer-gen/1.3

The default adapter is the offline stub from Chapter 5, so the whole loop runs
with no API key and no spend. Point ``--adapter`` at a hosted model to run it
for real.

Exit codes
----------
    0   pass, or escalate_for_review (a soft gate does not block a deployment)
    1   block_deployment: a hard gate fired
    2   usage or validation error, including an invalid trigger or model version

Pass ``--fail-on escalate`` where a team wants an escalation to fail the CI job
as well; the default follows the chapter, where the soft tier opens a review
without stopping the pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CH09 = _REPO_ROOT / "ch09"
sys.path.insert(0, str(_CH09))

DEFAULT_TRIGGERS_PATH = _CH09 / "triggers.yaml"
DEFAULT_GATES_PATH = _CH09 / "gates.yaml"

# The two evaluation scopes the chapter names, plus the cheapest tier from the
# data-trigger listing. A --trigger value is valid if it is one of these or the
# name of an event declared in ch09/triggers.yaml, which resolves to one.
TRIGGER_TYPES = ("full_benchmark", "targeted_benchmark", "metric_recalculation")

# Chapter 8 built the pilot as soccer_pilot_v1. In this repository that scenario
# library is the soccer_player set shipped with fairbench_genai, so the book's
# identifier is bound to it here rather than duplicating the YAML.
SCENARIO_ALIASES = {
    "soccer_pilot_v1": "soccer_player",
}

_MODELS_URI = re.compile(r"^models:/[\w.\-]+/[\w.\-]+$")
_REGISTRY_URI = re.compile(r"^[a-z][a-z0-9+.\-]*://\S+$")
_PROVIDER_VERSION = re.compile(r"^[A-Za-z][\w.\-]{2,}$")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")

# The dimensions Chapter 8 scoped for the soccer pilot, and how each one is
# carried through the library.
#
#   detected_key         the entity the vision classifier reports per image;
#                        RSI, ODE and SAR read the represented population from it
#   counterfactual_attr  the prompt-side attribute family CDS pairs against
#
# The second dimension is a documented approximation. The chapter calls it skin
# tone, the classifier reports a skin-tone label, and the soccer scenario's
# counterfactual family for it is race. CDS_skin_tone is therefore computed over
# the race counterfactual pairs, which is the closest prompt-side handle the
# scenario library offers. The gap is real and is recorded in ch09/CHAPTER_MAP.md.
DIMENSIONS = {
    "gender": {"detected_key": "gender", "counterfactual_attr": "gender"},
    "skin_tone": {"detected_key": "skin_tone", "counterfactual_attr": "race"},
}


class UsageError(Exception):
    """An argument the workflow passed does not make sense. Exit code 2."""


# --------------------------------------------------------------------------
# Argument validation
# --------------------------------------------------------------------------


def load_trigger_events(path: str | Path = DEFAULT_TRIGGERS_PATH) -> dict[str, str]:
    """Map each event name declared in triggers.yaml to its trigger_type."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    events: dict[str, str] = {}
    for block in data.values():
        if not isinstance(block, list):
            continue
        for entry in block:
            if not isinstance(entry, dict):
                continue
            name = entry.get("event") or entry.get("label")
            trigger_type = entry.get("trigger_type")
            if name and trigger_type:
                events[str(name)] = str(trigger_type)
    return events


def resolve_trigger(trigger: str, triggers_path: str | Path = DEFAULT_TRIGGERS_PATH) -> str:
    """Resolve ``--trigger`` to one of the three evaluation scopes.

    Accepts a scope directly (``full_benchmark``) or the name of an event
    declared in triggers.yaml (``model_version_change``), which resolves to the
    scope that event's policy assigns it.
    """
    if not trigger:
        raise UsageError("--trigger is required")
    if trigger in TRIGGER_TYPES:
        return trigger
    events = load_trigger_events(triggers_path)
    if trigger in events:
        return events[trigger]
    known = ", ".join([*TRIGGER_TYPES, *sorted(events)])
    raise UsageError(f"unknown trigger {trigger!r}. Declared triggers: {known}")


def validate_model_version(model_version: str) -> str:
    """Check that the version string identifies a model artifact, not a commit."""
    value = (model_version or "").strip()
    if not value:
        raise UsageError(
            "--model-version is required. Every run has to name the model artifact "
            "it evaluated, or the result cannot be attributed later."
        )
    if _COMMIT_SHA.match(value):
        raise UsageError(
            f"--model-version {value!r} looks like a commit SHA. A commit identifies the "
            "pipeline code, not the model that was evaluated; pass a model-registry URI "
            "(models:/soccer-gen/1.3) or a provider version string instead."
        )
    if _MODELS_URI.match(value) or _REGISTRY_URI.match(value) or _PROVIDER_VERSION.match(value):
        return value
    raise UsageError(
        f"--model-version {value!r} is not a recognizable model-registry URI or provider "
        "version string. Expected something like 'models:/soccer-gen/1.3', "
        "'registry://models/soccer-gen@1.3', or 'gpt-image-1-2026-03-01'."
    )


def resolve_scenario(scenario: str) -> str:
    """Resolve ``--scenario`` to a scenario-set name or a path to a scenario file."""
    if not scenario:
        raise UsageError("--scenario is required")
    if Path(scenario).exists():
        return scenario
    return SCENARIO_ALIASES.get(scenario, scenario)


# --------------------------------------------------------------------------
# Running the benchmark through fairbench_genai
# --------------------------------------------------------------------------


def _build_adapter(adapter_name: str, save_images: str | None):
    """Select a generation adapter. The default is offline and costs nothing."""
    if adapter_name == "stub":
        from fairbench_genai.adapters.image.stub import StubImageAdapter

        return StubImageAdapter()
    if adapter_name in ("gpt-image-1", "dalle3", "dall-e-3", "dalle2", "dall-e-2"):
        from fairbench_genai.adapters.image.dalle import DALLEAdapter

        model = "gpt-image-1" if adapter_name in ("gpt-image-1", "dalle3", "dall-e-3") else "dall-e-2"
        return DALLEAdapter(model=model, save_dir=save_images)
    if adapter_name.startswith("sd:") or adapter_name.startswith("sd-local:"):
        from fairbench_genai.adapters.image.stable_diffusion import StableDiffusionAdapter

        backend = "local" if adapter_name.startswith("sd-local:") else "hf_api"
        hf_id = adapter_name.split(":", 1)[1]
        return StableDiffusionAdapter(model=hf_id, backend=backend, save_dir=save_images)
    raise UsageError(
        f"unknown adapter {adapter_name!r}. Use 'stub' (offline), 'gpt-image-1', or 'sd:<hf-id>'."
    )


def _scenario_file_for(name: str) -> Path | None:
    """Locate the built-in scenario YAML for a scenario-set name, if there is one."""
    import fairbench_genai

    root = Path(fairbench_genai.__file__).resolve().parent / "scenarios" / "image"
    candidate = root / f"{name}.yaml"
    return candidate if candidate.exists() else None


async def _run_benchmark(scenario: str, adapter_name: str, concurrency: int, save_images: str | None):
    """Run the library's image benchmark and return the completed run object."""
    from fairbench_genai.core.image_engine import ImageBenchEngine
    from fairbench_genai.evaluation.image.stub import StubCLIPEvaluator, StubVisionAnalyzer
    from fairbench_genai.scenarios.registry import ScenarioRegistry

    engine = ImageBenchEngine()

    # The library's scenario registry is a process-wide singleton, and
    # registering the same set into it twice is an error. A CI entry point that
    # can be called more than once in one process (as the tests do) needs its
    # own registry, so this builds a fresh one with the same built-in sets
    # loaded and leaves the shared instance untouched.
    registry = ScenarioRegistry()
    registry.load_builtin()
    engine.scenario_registry = registry

    if Path(scenario).exists():
        loaded = registry.load_file(scenario)
        scenario_name = loaded.name
    else:
        path = _scenario_file_for(scenario)
        if path is not None:
            registry.load_file(str(path))
        scenario_name = scenario

    if adapter_name == "stub":
        # The offline stubs replay a recorded demographic reading instead of
        # calling a vision model, so the whole pipeline runs with no API key.
        vision: Any = StubVisionAnalyzer()
        clip: Any = StubCLIPEvaluator()
    else:
        from fairbench_genai.evaluation.image.clip_evaluator import CLIPEvaluator
        from fairbench_genai.evaluation.image.vision_analyzer import VisionAnalyzer

        vision = VisionAnalyzer()
        clip = CLIPEvaluator()

    return await engine.evaluate(
        model=_build_adapter(adapter_name, save_images),
        scenarios=[scenario_name],
        vision_analyzer=vision,
        clip_evaluator=clip,
        concurrency=concurrency,
    )


# --------------------------------------------------------------------------
# Projecting the library's metrics onto the Chapter 9 gate names
# --------------------------------------------------------------------------
#
# fairbench_genai computes one value per metric over the whole run: RSI, ODE,
# CDS, HSI, SAR, DSI. The Chapter 9 gate policy judges nine values, because RSI,
# ODE and CDS are gated per demographic dimension and SAR only on gender. The
# two views are reconciled here, in the shim, by re-running the library's own
# metric objects over per-dimension views of the same evaluated outputs. No
# metric is reimplemented and nothing in src/ changes.


def _dimension_view(outputs: list[Any], detected_key: str) -> list[Any]:
    """Outputs seen through one classifier dimension.

    Each output keeps only the detected entity for this dimension, and its
    counterfactual value is set to the dimension value the classifier read from
    the image. That is what lets RSI (detected mode), ODE (attribute counts) and
    SAR (observed against baseline) each report on one dimension at a time.
    """
    from fairbench_genai.core.types import EvaluatedOutput

    view = []
    for output in outputs:
        values = output.detected_entities.get(detected_key)
        if not values:
            continue
        view.append(
            EvaluatedOutput(
                **{
                    **output.model_dump(),
                    "detected_entities": {detected_key: list(values)},
                    "counterfactual_attribute": detected_key,
                    "counterfactual_value": values[0],
                }
            )
        )
    return view


def _counterfactual_view(outputs: list[Any], attribute: str) -> list[Any]:
    """Outputs restricted to one counterfactual family, for CDS.

    CDS pairs each counterfactual output against the base output of its
    scenario, so the view keeps the base outputs and only those counterfactuals
    that vary the attribute in question.
    """
    return [
        output
        for output in outputs
        if (not output.is_counterfactual) or output.counterfactual_attribute == attribute
    ]


def _safe(compute) -> float | None:
    """Run a metric, returning None where it has nothing to work on.

    A metric with no usable samples is reported as not evaluated rather than as
    a zero. The gate evaluator prints those separately, so a dimension the run
    could not measure is visible instead of passing quietly.
    """
    try:
        result = compute()
    except Exception:  # noqa: BLE001 - an unmeasurable dimension is not a crash
        return None
    value = float(result.value)
    if value != value:  # NaN
        return None
    return value


def project_metrics(run: Any) -> dict[str, float | None]:
    """Compute the nine gate metrics from a completed library run."""
    from fairbench_genai.metrics.cds import CounterfactualDivergenceScore
    from fairbench_genai.metrics.dsi import DifferentialServiceIndex
    from fairbench_genai.metrics.hsi import HarmSeverityIndex
    from fairbench_genai.metrics.ode import OutputDiversityEntropy
    from fairbench_genai.metrics.rsi import RepresentationSkewIndex
    from fairbench_genai.metrics.sar import StereotypeAmplificationRatio

    outputs = [image.to_evaluated_output() for image in run.evaluated_images]

    rsi = RepresentationSkewIndex(attribute_extractor="detected")
    ode = OutputDiversityEntropy(diversity_method="attribute_counts")
    cds = CounterfactualDivergenceScore()
    sar = StereotypeAmplificationRatio()

    metrics: dict[str, float | None] = {}
    for dimension, spec in DIMENSIONS.items():
        detected = _dimension_view(outputs, spec["detected_key"])
        pairs = _counterfactual_view(outputs, spec["counterfactual_attr"])
        metrics[f"RSI_{dimension}"] = _safe(lambda d=detected: rsi.compute(d))
        metrics[f"ODE_{dimension}"] = _safe(lambda d=detected: ode.compute(d))
        metrics[f"CDS_{dimension}"] = _safe(lambda p=pairs: cds.compute(p))

    # SAR is recorded for gender alone. It needs a real-world base rate to
    # compare against, and gender is the only dimension in this scenario with a
    # usable one, so reporting SAR_skin_tone would be a number with nothing
    # behind it. Chapter 7 develops that selection rule.
    gender_view = _dimension_view(outputs, DIMENSIONS["gender"]["detected_key"])
    metrics["SAR_gender"] = _safe(lambda: sar.compute(gender_view))

    # HSI and DSI are dimension-agnostic: harm severity and refusal rate are
    # properties of the run, so they come straight from the library.
    metrics["HSI"] = _safe(lambda: HarmSeverityIndex().compute(outputs))
    metrics["DSI"] = _safe(lambda: DifferentialServiceIndex().compute(outputs))
    return metrics


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fairbench.run",
        description=(
            "Run a fairness benchmark and evaluate it against the Chapter 9 gate "
            "policy. Exits 1 when a hard gate blocks the deployment."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trigger",
        required=True,
        help=(
            "What caused this run: an evaluation scope (full_benchmark, "
            "targeted_benchmark, metric_recalculation) or an event name declared "
            "in ch09/triggers.yaml."
        ),
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario set name (e.g. soccer_pilot_v1) or a path to a scenario YAML.",
    )
    parser.add_argument(
        "--model-version",
        required=True,
        help=(
            "The model artifact this run evaluated: a registry URI "
            "(models:/soccer-gen/1.3) or a provider version string. Not optional, "
            "and not a commit SHA."
        ),
    )
    parser.add_argument(
        "--adapter",
        default="stub",
        help="Generation adapter: 'stub' (offline, the default), 'gpt-image-1', 'sd:<hf-id>'.",
    )
    parser.add_argument("--gates", default=str(DEFAULT_GATES_PATH), help="Gate policy YAML.")
    parser.add_argument(
        "--exceptions", default=None, help="Exception log YAML to apply to soft-gate breaches."
    )
    parser.add_argument(
        "--triggers", default=str(DEFAULT_TRIGGERS_PATH), help="Trigger policy YAML."
    )
    parser.add_argument("--output", default=None, help="Write the result document to this path.")
    parser.add_argument("--save-images", default=None, help="Directory to save generated images.")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent generations.")
    parser.add_argument(
        "--fail-on",
        choices=("block", "escalate"),
        default="block",
        help="Exit non-zero on a block (default) or on any escalation as well.",
    )
    parser.add_argument("--json", action="store_true", help="Print the decision as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from gate_evaluator import GatePolicy  # noqa: PLC0415 - ch09/ is on sys.path
    from exception_record import load_exception_log  # noqa: PLC0415

    try:
        trigger_type = resolve_trigger(args.trigger, args.triggers)
        model_version = validate_model_version(args.model_version)
        scenario = resolve_scenario(args.scenario)
        policy = GatePolicy.load(args.gates)
        exceptions = (
            load_exception_log(args.exceptions, policy.hard_gate_metrics)
            if args.exceptions
            else None
        )
    except UsageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - a bad policy file is a usage error too
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    print(f"trigger       : {args.trigger} -> {trigger_type}")
    print(f"scenario      : {args.scenario} -> {scenario}")
    print(f"model version : {model_version}")
    print(f"adapter       : {args.adapter}")
    print()

    try:
        run = asyncio.run(
            _run_benchmark(scenario, args.adapter, args.concurrency, args.save_images)
        )
    except Exception as e:  # noqa: BLE001 - a failed run is a pipeline failure
        print(f"benchmark failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    metrics = project_metrics(run)
    report = policy.evaluate(metrics, exceptions)

    document: dict[str, Any] = {
        "run_id": str(run.id),
        "model_version": model_version,
        "scenario_id": args.scenario,
        "trigger_type": trigger_type,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "gate_decision": report.to_dict(),
    }

    print()
    if args.json:
        print(json.dumps(document, indent=2, default=str))
    else:
        print(report.summary())

    if args.output:
        Path(args.output).write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
        )
        print(f"\nresult written to {args.output}")

    if report.decision == "block_deployment":
        return 1
    if args.fail_on == "escalate" and report.decision == "escalate_for_review":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
