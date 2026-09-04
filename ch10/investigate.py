"""Run a fairness-regression investigation from the command line.

Book: Chapter 10, "Traces as the debugging backbone" and "Measurement quality
metrics".

    python ch10/investigate.py --list
    python ch10/investigate.py classifier-shift
    python ch10/investigate.py model-regression
    python ch10/investigate.py --current soccer_2026_w14 --prior soccer_2026_w13 \
        --metric RSI_gender --partition role=goalkeeper
    python ch10/investigate.py --quality ch10/fixtures/measurement_quality_w14.yaml
    python ch10/investigate.py --quality-from-traces --current soccer_2026_w14 \
        --prior soccer_2026_w13

The two modes exit differently, and the difference is deliberate. An
investigation always exits zero, because it is a diagnosis and not a decision:
it says what the evidence supports and leaves the deployment question to the
gates of Chapter 9. A measurement-quality check exits non-zero when a target is
breached, so it can gate a pipeline, because a benchmark whose own measurement
quality has degraded should stop feeding decisions until someone has looked.

Everything here runs against the fixtures in ``ch10/fixtures/``, so it needs no
API key, no network, and no configuration.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from measurement_quality import (  # noqa: E402
    evaluate,
    evaluate_report,
    measure_from_traces,
    summarize,
)
from regression_report import RegressionReport, investigate_regression  # noqa: E402
from trace_store import TraceStore  # noqa: E402

WIDTH = 78

# The two worked investigations the fixtures carry. Both are stored regressions
# of the same size and direction; what separates them is which layer moved.
SCENARIOS = {
    "classifier-shift": {
        "metric": "RSI_gender",
        "current": "soccer_2026_w14",
        "prior": "soccer_2026_w13",
        "blurb": (
            "A new perception classifier version ships between the two runs. The model "
            "version does not change, so every output is served from cache."
        ),
    },
    "model-regression": {
        "metric": "RSI_gender",
        "current": "soccer_2026_w21",
        "prior": "soccer_2026_w20",
        "blurb": (
            "The model version changes between the two runs while the classifier is held "
            "fixed, so the outputs are regenerated and the probe sees the difference."
        ),
    },
}


def _wrap(text: str, indent: str = "  ") -> str:
    return "\n".join(
        textwrap.wrap(text, width=WIDTH, initial_indent=indent, subsequent_indent=indent)
    )


def _fmt(value: float, places: int = 4) -> str:
    return "n/a" if value != value else f"{value:.{places}f}"


def render(report: RegressionReport, top: int = 8) -> str:
    """Render a regression report as the text an on-call engineer reads first."""
    delta = report.partitioned_delta
    dist = report.distribution_shift
    conf = report.classifier_confidence_shift
    lines: list[str] = []

    lines.append("=" * WIDTH)
    lines.append(f"REGRESSION INVESTIGATION  {report.metric}")
    partition = report.attribute_partition or "whole scenario"
    lines.append(f"{report.prior_run_id}  ->  {report.current_run_id}   partition: {partition}")
    lines.append("=" * WIDTH)

    lines.append("")
    lines.append("LAYER 1  aggregate: did it move, and where?")
    lines.append(
        f"  {report.metric}  {_fmt(delta.prior_value)}  ->  {_fmt(delta.current_value)}"
        f"   delta {delta.delta:+.4f}"
    )
    ci_low, ci_high = delta.delta_confidence_interval
    test = "paired t-test" if delta.paired else "Welch t-test (prompt libraries differ)"
    lines.append(
        f"  95% CI [{_fmt(ci_low)}, {_fmt(ci_high)}]   p={_fmt(delta.p_value)}  ({test})"
    )
    lines.append(
        f"  scored records {delta.n_prior} -> {delta.n_current}"
        f"   excluded {delta.prior_exclusion_rate:.1%} -> {delta.current_exclusion_rate:.1%}"
    )
    if delta.unmeasured_prompts_current or delta.unmeasured_prompts_prior:
        lines.append(
            f"  prompts with no scored record: prior {len(delta.unmeasured_prompts_prior)},"
            f" current {len(delta.unmeasured_prompts_current)}"
        )
    lines.append("  by prompt cluster:")
    for slice_ in delta.by_cluster:
        lines.append(
            f"    {slice_.partition:<16} {_fmt(slice_.prior_value, 3)}"
            f" -> {_fmt(slice_.current_value, 3)}"
            f"   delta {slice_.delta:+.3f}   share of movement {slice_.share_of_movement:+.0%}"
        )
    localized = delta.localized_clusters
    if localized:
        lines.append(
            _wrap(
                "The movement is carried by " + ", ".join(localized) + ". A cause specific to "
                "these prompts is more likely than one upstream of the whole scenario.",
                indent="    ",
            )
        )

    lines.append("")
    lines.append("LAYER 2  distributional: did the model's own outputs move?")
    lines.append(f"  reference probe: {dist.reference_probe}  (pinned across runs)")
    for category in sorted(set(dist.current_distribution) | set(dist.prior_distribution)):
        lines.append(
            f"    {category:<16} {dist.prior_distribution.get(category, 0.0):.3f}"
            f" -> {dist.current_distribution.get(category, 0.0):.3f}"
            f"   {dist.category_deltas.get(category, 0.0):+.3f}"
        )
    lines.append(
        f"  Jensen-Shannon distance {_fmt(dist.js_distance)}"
        f"   chi-square p={_fmt(dist.chi2_p_value)}"
        f"   -> {'SHIFTED' if dist.shifted else 'no shift detected'}"
    )

    lines.append("")
    lines.append("GUARD    classifier confidence: did the instrument move instead?")
    lines.append(
        f"  versions  {', '.join(conf.prior_versions) or 'none'}  ->  "
        f"{', '.join(conf.current_versions) or 'none'}"
        f"   {'CHANGED' if conf.version_changed else 'unchanged'}"
    )
    lines.append(
        f"  mean confidence   {_fmt(conf.prior_mean_confidence, 3)} -> "
        f"{_fmt(conf.current_mean_confidence, 3)}   ({conf.mean_delta:+.3f})"
    )
    lines.append(
        f"  borderline rate   {conf.prior_borderline_rate:.1%} -> "
        f"{conf.current_borderline_rate:.1%}   ({conf.borderline_rate_delta:+.1%})"
    )
    lines.append(
        f"  KS statistic {_fmt(conf.ks_statistic, 3)}   p={_fmt(conf.ks_p_value)}"
        f"   -> {'SHIFTED' if conf.shifted else 'no shift detected'}"
    )

    lines.append("")
    lines.append(f"LAYER 3  instance: which prompts carried it?  (top {top} of "
                 f"{len(report.top_contributing_prompts)} reported)")
    lines.append(
        f"    {'prompt_id':<16} {'cluster':<14} {'prior':>6} {'curr':>6} "
        f"{'contrib':>8} {'share':>7} {'conf':>7} {'excl':>9}"
    )
    for row in report.top_contributing_prompts[:top]:
        lines.append(
            f"    {row.prompt_id:<16} {row.prompt_cluster:<14} {_fmt(row.prior_value, 3):>6} "
            f"{_fmt(row.current_value, 3):>6} {row.contribution:>+8.4f} "
            f"{row.share_of_movement:>+7.0%} {row.confidence_delta:>+7.3f} "
            f"{row.excluded_prior:>4}->{row.excluded_current:<4}"
        )

    lines.append("")
    lines.append("=" * WIDTH)
    lines.append(f"VERDICT  {report.verdict.value}")
    lines.append(_wrap(report.recommended_next_step))
    lines.append("=" * WIDTH)
    return "\n".join(lines)


def _list_runs() -> int:
    store = TraceStore()
    runs = store.runs()
    if not runs:
        print(f"no trace fixtures under {store.root}; run: python ch10/make_fixtures.py")
        return 1
    print(f"trace store: {store.root}\n")
    for run_id in runs:
        records = store.read(run_id)
        metrics = sorted({r.metric for r in records})
        models = sorted({r.model_call.model_version for r in records})
        classifiers = sorted(
            {f"{c.classifier}@{c.classifier_version}" for r in records for c in r.classifier_calls}
        )
        print(f"  {run_id}")
        print(f"      {len(records)} traces   metrics: {', '.join(metrics)}")
        print(f"      model: {', '.join(models)}   classifiers: {', '.join(classifiers)}")
    print("\nworked investigations:")
    for name, spec in SCENARIOS.items():
        print(f"  {name:<20} {spec['prior']} -> {spec['current']}  ({spec['metric']})")
        print(_wrap(spec["blurb"], indent="      "))
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="investigate.py",
        description="Walk a fairness regression through the aggregate, distributional, "
        "and instance layers against the Chapter 10 trace fixtures.",
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        choices=sorted(SCENARIOS),
        help="one of the worked investigations the fixtures carry",
    )
    parser.add_argument("--metric", default="RSI_gender", help="metric to investigate")
    parser.add_argument("--current", help="run the alert fired on")
    parser.add_argument("--prior", help="run to compare against")
    parser.add_argument(
        "--partition",
        default=None,
        help="restrict to a slice, e.g. 'role=goalkeeper' or 'skin_tone_band=IV-VI,locale=in'",
    )
    parser.add_argument("--top", type=int, default=8, help="rows of the instance layer to print")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--list", action="store_true", help="list the runs in the trace store")
    parser.add_argument("--quality", metavar="REPORT", help="evaluate a measurement-quality report")
    parser.add_argument(
        "--quality-from-traces",
        action="store_true",
        help="compute a measurement-quality report from the two runs and evaluate it",
    )
    args = parser.parse_args(argv)

    if args.list:
        return _list_runs()

    if args.quality:
        assessment = evaluate_report(args.quality)
        print(summarize(assessment, width=WIDTH))
        return 0 if assessment.passed else 1

    if args.scenario:
        spec = SCENARIOS[args.scenario]
        metric = spec["metric"] if args.metric == "RSI_gender" else args.metric
        current, prior = spec["current"], spec["prior"]
    else:
        metric, current, prior = args.metric, args.current, args.prior
        if not (current and prior):
            parser.error("give a scenario name, or both --current and --prior")

    if args.quality_from_traces:
        observed, provenance = measure_from_traces(current, prior, metric)
        assessment = evaluate(
            observed,
            report_id=f"mq_{current}",
            scenario="soccer_pilot_v1",
            metric=metric,
            notes=provenance,
        )
        print(summarize(assessment, width=WIDTH))
        return 0 if assessment.passed else 1

    report = investigate_regression(metric, current, prior, args.partition)
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print(render(report, top=args.top))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except BrokenPipeError:
        # Piping into head or less closes the stream early; that is not an error.
        sys.stderr.close()
        raise SystemExit(0) from None
