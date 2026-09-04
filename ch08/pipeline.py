"""The evaluation pipeline: classification, metric computation, and the summary.

Chapter 5 drew four boxes. This module is three of them wired together -
classifier, metric engine, and the summary the artifact store writes - with the
model access layer in ``model_access.py`` feeding the first one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from classifiers import FitzpatrickClassifier, GenderClassifier, load_metadata
from metrics import (  # noqa: I001 - grouped by role, not alphabetically
    AMBIGUOUS,
    CDS,
    DSI,
    HSI,
    ODE,
    RSI,
    SAR,
    AttributeDistribution,
    MetricResult,
    label_of,
)
from thresholds import ThresholdEvaluator

CONFIG_DIR = Path(__file__).resolve().parent / "config"


# ---------------------------------------------------------------- classify --


def classify_outputs(
    image_dir: str | Path,
    classifier_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Run both classifiers over every image and collect the labels.

    One record per image, carrying the metadata that came with it, the two
    labels, and the harm and service observations the metric engine needs later.
    """
    image_dir = Path(image_dir)
    gender_clf = GenderClassifier(
        model=classifier_config["gender"]["model"],
        confidence_threshold=classifier_config["gender"].get("confidence_threshold", 0.7),
    )
    skin_clf = FitzpatrickClassifier(
        model=classifier_config["skin_tone"]["model"],
        confidence_threshold=classifier_config["skin_tone"].get("confidence_threshold", 0.65),
    )
    taxonomy = {
        "gender": list(classifier_config["gender"].get("categories", [])),
        "skin_tone": list(classifier_config["skin_tone"].get("categories", [])),
    }

    results: list[dict[str, Any]] = []
    # Individual processing is slow at scale; a production run batches these.
    for image_path in sorted(image_dir.glob("*.png")):
        metadata = load_metadata(image_path)
        results.append(
            {
                "image_id": image_path.stem,
                "metadata": metadata,
                "taxonomy": taxonomy,
                "gender": gender_clf.classify(image_path),
                "skin_tone": skin_clf.classify(image_path),
                "harm_scores": metadata.get("harm_scores", {}),
                "service": metadata.get("service", {}),
            }
        )
    return results


def ambiguous_count(classifier_results: Sequence[Mapping[str, Any]]) -> int:
    """Images either classifier could not confidently place.

    Counted rather than discarded. A rate above about 5% says the prompts or the
    thresholds need revisiting before any metric built on these labels is read.
    """
    return sum(
        1
        for r in classifier_results
        if AMBIGUOUS in (label_of(r, "gender"), label_of(r, "skin_tone"))
    )


# ------------------------------------------------------------- selection ----


def filter_by_modifier(
    classifier_results: Sequence[Mapping[str, Any]],
    modifier: str | None,
) -> list[Mapping[str, Any]]:
    """Select the outputs produced under one prompt condition.

    ``None`` selects the neutral prompts, the ones that named no attribute at
    all, which are the baseline every counterfactual is compared against.
    """
    return [r for r in classifier_results if r["metadata"].get("modifier") == modifier]


def extract_distribution(
    classifier_results: Sequence[Mapping[str, Any]],
    attribute: Any,
    categories: Sequence[str] | None = None,
) -> AttributeDistribution:
    """Build the label counts for one attribute, split by what the prompt named.

    Two counts come back in one object because RSI and ODE ask different
    questions. ``unnamed`` holds the outputs whose prompt said nothing about this
    attribute - the neutral prompts plus the counterfactuals that named the other
    one - and is what RSI reads, since it is the only evidence about the model's
    default. ``overall`` holds every output in the run and is what ODE reads,
    since coverage of the taxonomy is a property of the run as a whole.

    ``attribute`` accepts either a name or a scenario's attribute object. When a
    name is given, the declared taxonomy is recovered from the records
    themselves, which carry the label space their classifier was drawn from.
    """
    name = getattr(attribute, "name", attribute)
    if categories is None:
        categories = getattr(attribute, "categories", None)
    if categories is None:
        for record in classifier_results:
            declared = (record.get("taxonomy") or {}).get(name)
            if declared:
                categories = list(declared)
                break
    if not categories:
        # Last resort: the categories the run happened to produce. This hides
        # erasure, so it is a fallback and never the intended path.
        categories = sorted(
            {label_of(r, name) for r in classifier_results} - {AMBIGUOUS}
        )

    overall: dict[str, int] = {}
    unnamed: dict[str, int] = {}
    amb_overall = amb_unnamed = 0
    for record in classifier_results:
        label = label_of(record, name)
        prompt_named_this = record["metadata"].get("attribute") == name
        if label == AMBIGUOUS:
            amb_overall += 1
            if not prompt_named_this:
                amb_unnamed += 1
            continue
        overall[label] = overall.get(label, 0) + 1
        if not prompt_named_this:
            unnamed[label] = unnamed.get(label, 0) + 1

    return AttributeDistribution(
        attribute=str(name),
        categories=list(categories),
        overall=overall,
        unnamed=unnamed,
        ambiguous_overall=amb_overall,
        ambiguous_unnamed=amb_unnamed,
    )


# ---------------------------------------------------------------- metrics ---


def _service_records(classifier_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reshape the run's service observations into what DSI expects."""
    out: list[dict[str, Any]] = []
    for record in classifier_results:
        service = record.get("service") or {}
        out.append(
            {
                "group": record["metadata"].get("condition", "neutral"),
                "refused": bool(service.get("refused", False)),
                "images_delivered": service.get("images_delivered", 1),
                "helpfulness": service.get("helpfulness"),
            }
        )
    return out


def load_baselines(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else CONFIG_DIR / "baselines.yaml"
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def compute_metrics(
    classifier_results: Sequence[Mapping[str, Any]],
    scenario: Any,
    baselines: Mapping[str, Any] | None = None,
) -> dict[str, MetricResult]:
    """Compute the full metric suite from the classifier labels.

    RSI for the skew, CDS for the implicit prior underneath it, and ODE for how
    narrow the range of representations has become; then SAR, HSI and DSI, which
    the run artifact reports and Chapter 7 defines.
    """
    metrics: dict[str, MetricResult] = {}
    distributions: dict[str, AttributeDistribution] = {}

    for attribute in scenario.sensitive_attributes:
        observed = extract_distribution(classifier_results, attribute.name)
        distributions[attribute.name] = observed
        reference = attribute.reference_distribution
        metrics[f"RSI_{attribute.name}"] = RSI.compute(
            observed=observed,
            reference=reference,
        )
        metrics[f"ODE_{attribute.name}"] = ODE.compute(
            observed=observed,
        )

    for attr in scenario.sensitive_attributes:
        per_modifier: list[float] = []
        for modifier in attr.counterfactual_modifiers:
            result = CDS.compute(
                neutral_results=filter_by_modifier(classifier_results, None),
                specified_results=filter_by_modifier(classifier_results, modifier),
            )
            metrics[f"CDS_{attr.name}_{modifier}"] = result
            per_modifier.append(result.value)
        # The scorecard reports one CDS per attribute: the mean of the
        # per-modifier scores, which is the "mean divergence across groups" the
        # metric specification asks for.
        metrics[f"CDS_{attr.name}"] = MetricResult(
            name=f"CDS_{attr.name}",
            value=sum(per_modifier) / len(per_modifier),
            details={
                "attribute": attr.name,
                "per_modifier": {
                    m: metrics[f"CDS_{attr.name}_{m}"].value
                    for m in attr.counterfactual_modifiers
                },
                "aggregation": "mean over counterfactual modifiers",
            },
        )

    baselines = baselines if baselines is not None else load_baselines()
    for attr_name, spec in baselines.items():
        if attr_name not in distributions:
            continue
        metrics[f"SAR_{attr_name}"] = SAR.compute(
            observed=distributions[attr_name],
            association_category=spec["association_category"],
            baseline_rate=float(spec["baseline_rate"]),
            baseline_source=str(spec.get("source", "unspecified")).strip(),
        )

    metrics["HSI"] = HSI.compute(classifier_results)
    metrics["DSI"] = DSI.compute(_service_records(classifier_results))
    return metrics


# ---------------------------------------------------------------- summary ---


def generate_summary(
    metrics: Mapping[str, Any],
    scenario: Any,
    run_id: str | None = None,
    model_id: str | None = None,
    evaluator: ThresholdEvaluator | None = None,
) -> str:
    """Render the metrics summary a product team reads.

    The values come from the metric engine. The band beside each one, and the
    ``summary`` line at the bottom, come from the threshold layer, which is a
    separate component on purpose: the pilot measures, and the recommendation is
    the threshold layer's reading of what was measured.
    """
    evaluator = evaluator or ThresholdEvaluator()
    verdict = evaluator.evaluate(metrics)

    lines: list[str] = []
    lines.append(f"run_id: {run_id or scenario.id}")
    lines.append(f"model: {model_id or 'unknown'}")
    lines.append(f"scenario: {scenario.id}")
    lines.append("metrics:")
    for item in verdict.metrics:
        label = f"{item.metric}:"
        lines.append(
            f"  {label:<16}{item.value:.2f}   # {item.band:<5} -- threshold {item.threshold:.2f}"
        )
    lines.append(f"summary: {verdict.summary}")
    lines.append("flags:")
    for flag in verdict.flags:
        lines.append(f"  - metric: {flag['metric']}")
        lines.append(f"    note: \"{flag['note']}\"")
    return "\n".join(lines) + "\n"


__all__ = [
    "ambiguous_count",
    "classify_outputs",
    "compute_metrics",
    "extract_distribution",
    "filter_by_modifier",
    "generate_summary",
    "load_baselines",
]
