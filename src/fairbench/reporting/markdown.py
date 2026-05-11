"""Markdown scorecard renderer for FAIRBench evaluation runs.

Produces a human-readable scorecard structured as:
  - Header: run metadata
  - Summary: metric scores table with band indicators
  - Details: one subsection per metric with data-driven reasoning
  - Known Limitations

Intended for model cards, audit reports, and human review.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fairbench.core.types import Distribution, EvaluationRun
from fairbench.metrics.base import Metric
from fairbench.metrics.cds import CounterfactualDivergenceScore
from fairbench.metrics.dsi import DifferentialServiceIndex
from fairbench.metrics.hsi import HarmSeverityIndex
from fairbench.metrics.ode import OutputDiversityEntropy
from fairbench.metrics.rsi import RepresentationSkewIndex
from fairbench.metrics.sar import StereotypeAmplificationRatio
from fairbench.reporting.scorecard import _band, generate_scorecard

# ---------------------------------------------------------------------------
# Band display helpers
# ---------------------------------------------------------------------------

_BAND_EMOJI = {
    "pass": "🟢 Pass",
    "watch": "🟡 Watch",
    "flag": "🔴 Flag",
    "fail": "⛔ Fail",
    "unknown": "⬜ Unknown",
}

_BAND_ACTION = {
    "pass": "Monitor; no immediate action",
    "watch": "Investigate scenario drivers",
    "flag": "Remediate before release",
    "fail": "Do not release; escalate",
    "unknown": "Review",
}

_METRIC_FULL_NAMES = {
    "RSI": "Representation Skew Index",
    "SAR": "Stereotype Amplification Ratio",
    "ODE": "Output Diversity Entropy",
    "CDS": "Counterfactual Divergence Score",
    "HSI": "Harm Severity Index",
    "DSI": "Differential Service Index",
}

_DEFAULT_METRICS: dict[str, Metric] = {
    "RSI": RepresentationSkewIndex(),
    "SAR": StereotypeAmplificationRatio(),
    "ODE": OutputDiversityEntropy(),
    "CDS": CounterfactualDivergenceScore(),
    "HSI": HarmSeverityIndex(),
    "DSI": DifferentialServiceIndex(),
}

_KNOWN_LIMITATIONS = [
    "**Binary classifiers:** Most demographic classifiers detect binary gender only; "
    "human review is required for non-binary representation assessment.",
    "**English-language focus:** Harm and demographic classifiers have reduced "
    "reliability for non-English outputs.",
    "**Classifier bias:** Harm classifiers trained on Western internet data may "
    "under-score harm in non-Western cultural contexts.",
    "**Single-turn evaluation:** All prompts are single-turn; multi-turn fairness "
    "requires separate scenario design.",
    "**SAR baselines:** SAR cannot be computed reliably without a defensible "
    "real-world baseline; scenarios without a baseline are omitted.",
    "**DSI helpfulness proxy:** The helpfulness component uses response length "
    "as a proxy unless human ratings are provided via `helpfulness_score`.",
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_markdown_scorecard(
    run: EvaluationRun,
    benchmark_name: str | None = None,
    baseline: Distribution | None = None,
    metrics: dict[str, Metric] | None = None,
) -> str:
    """Generate a Markdown scorecard for a completed evaluation run.

    Args:
        run: The completed evaluation run.
        benchmark_name: Optional human-readable benchmark name for the header.
        baseline: Optional baseline distribution passed through to metrics.
        metrics: Custom metric instances.  Defaults to all six built-in metrics.

    Returns:
        A Markdown string ready to write to a ``.md`` file.
    """
    metric_registry = metrics if metrics is not None else _DEFAULT_METRICS

    # Reuse the JSON scorecard computation for per-metric summary data
    scorecard_data = generate_scorecard(run, baseline=baseline, metrics=metric_registry)
    summary = scorecard_data.get("summary", {})

    # Build per-metric reasoning via each metric's interpret() method
    reasoning: dict[str, str] = {}
    for metric_name, metric in metric_registry.items():
        # Find the run-level MetricResult for this metric
        run_result = next(
            (mr for mr in run.metric_results if mr.metric_name == metric_name),
            None,
        )
        if run_result is not None:
            try:
                reasoning[metric_name] = metric.interpret(run_result)
            except Exception:
                reasoning[metric_name] = summary.get(metric_name, {}).get("band", "")

    name = benchmark_name or run.config_snapshot.get("benchmark_name", "FAIRBench Audit")
    model = run.model_info
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sections: list[str] = []

    # --- Header ---
    sections.append(_render_header(name, model, run, generated_at))

    # --- Summary ---
    sections.append(_render_summary(summary, run.metrics_requested))

    # --- Details ---
    sections.append(_render_details(summary, reasoning, run.metrics_requested, metric_registry))

    # --- Known Limitations ---
    sections.append(_render_limitations())

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_header(
    name: str,
    model: Any,
    run: EvaluationRun,
    generated_at: str,
) -> str:
    lines = [
        f"# FAIRBench Scorecard",
        f"",
        f"**Benchmark:** {name}  ",
        f"**Model:** {model.name} ({model.provider})  ",
        f"**Run ID:** `{run.id}`  ",
        f"**Generated:** {generated_at}  ",
        f"**Scenarios:** {', '.join(run.scenario_sets)}  ",
        f"**Metrics computed:** {', '.join(run.metrics_requested)}  ",
    ]
    return "\n".join(lines)


def _render_summary(
    summary: dict[str, Any],
    metrics_requested: list[str],
) -> str:
    """Render the summary table — metric scores, bands, and recommended actions."""
    lines = [
        "---",
        "",
        "## Summary",
        "",
        "| Metric | Full Name | Score | Band | Recommended Action |",
        "|--------|-----------|------:|------|-------------------|",
    ]

    worst_band = "pass"
    band_rank = {"pass": 0, "watch": 1, "flag": 2, "fail": 3, "unknown": -1}

    for metric_name in metrics_requested:
        if metric_name not in summary:
            continue
        entry = summary[metric_name]
        score = entry.get("score", float("nan"))
        band = entry.get("band", "unknown")
        full_name = _METRIC_FULL_NAMES.get(metric_name, metric_name)
        emoji = _BAND_EMOJI.get(band, "⬜")
        action = _BAND_ACTION.get(band, "Review")

        score_str = f"{score:.3f}" if score == score else "N/A"  # NaN check
        lines.append(f"| {metric_name} | {full_name} | {score_str} | {emoji} | {action} |")

        if band_rank.get(band, -1) > band_rank.get(worst_band, 0):
            worst_band = band

    lines.append("")

    # Overall verdict line
    verdict_emoji = _BAND_EMOJI.get(worst_band, "⬜")
    action_text = _BAND_ACTION.get(worst_band, "Review")
    lines.append(f"**Overall verdict: {verdict_emoji}** — {action_text}.")

    return "\n".join(lines)


def _render_details(
    summary: dict[str, Any],
    reasoning: dict[str, str],
    metrics_requested: list[str],
    metric_registry: dict[str, Metric],
) -> str:
    """Render the Details section — one subsection per metric with reasoning."""
    lines = [
        "---",
        "",
        "## Details",
    ]

    for metric_name in metrics_requested:
        if metric_name not in summary:
            continue
        entry = summary[metric_name]
        score = entry.get("score", float("nan"))
        band = entry.get("band", "unknown")
        full_name = _METRIC_FULL_NAMES.get(metric_name, metric_name)
        emoji = _BAND_EMOJI.get(band, "⬜")
        score_str = f"{score:.3f}" if score == score else "N/A"
        n = entry.get("n_scenarios", 0)

        lines.append("")
        lines.append(f"### {metric_name} — {full_name}")
        lines.append(f"**Score: {score_str} | {emoji}** *(across {n} scenario(s))*")
        lines.append("")

        reason = reasoning.get(metric_name, "")
        if reason:
            # Split on "  " double-space separators that we use in interpret() to join sentences
            sentences = [s.strip() for s in reason.split("  ") if s.strip()]
            lines.extend(sentences)
        else:
            lines.append("_No reasoning available — insufficient output data._")

        lines.append("")
        lines.append("---")

    return "\n".join(lines)


def _render_limitations() -> str:
    lines = ["## Known Limitations", ""]
    for lim in _KNOWN_LIMITATIONS:
        lines.append(f"- {lim}")
    return "\n".join(lines)
