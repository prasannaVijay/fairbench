"""Measurement-quality targets for the benchmarking system, and the checks behind them.

Book: Chapter 10, "Measurement quality metrics".

A metric is only as trustworthy as the process that measured it. The three
quantities in ``measurement_quality.yaml`` describe how much confidence a
reading deserves, independent of the reading itself: how much the metric moves
when nothing has changed, whether the scenario carries enough data to detect the
effect it claims to detect, and how much of a movement across a classifier
version boundary is attributable to the classifier rather than the model.

This module does two things. It parses the targets and checks an observed
measurement-quality report against them, reporting each breach with the action
the block prescribes. It also carries estimators for the three quantities, so
that a report can be produced from the trace store rather than assembled by
hand. The estimators are separated from the evaluator on purpose: a team with
its own measurement pipeline should be able to keep the targets and the
governance path while replacing the arithmetic.

Two honest limits on the estimators are worth stating before anyone reads a
number out of them.

``bootstrap_metric_sd`` resamples prompts within a single run. It captures the
variation that comes from which prompts and replicates happened to be drawn, and
it cannot see the variation that comes from running the generation API again on
another day. It is therefore a lower bound on the noise floor, and the target in
the block is defined against the real thing: the standard deviation across
independent replicate runs, which ``replicate_stability`` computes when those
runs exist. Substituting the bootstrap for the real measurement will make the
noise floor look better than it is.

``power_from_traces`` estimates power from the observed spread of a single pair
of runs. A power estimate computed after the fact from the same data it will be
used to judge is a weaker instrument than a power calculation done at design
time, which is exactly why the block sets its frequency to the scenario design
review and not to every run.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats
from trace_store import TraceSet, TraceStore, load_traces

DEFAULT_TARGETS_PATH = Path(__file__).resolve().parent / "measurement_quality.yaml"

# The effect size and significance level the block's power target is defined
# against. Both are stated in the description text; they are repeated here so
# the estimator and the target cannot drift apart silently.
POWER_EFFECT_SIZE = 0.05
POWER_ALPHA = 0.05

_TARGET_PATTERN = re.compile(r"^\s*(?P<op><=|>=|<|>)\s*(?P<value>[0-9]*\.?[0-9]+)\s*(?P<scope>.*)$")


class QualityTarget(BaseModel):
    """One measurement-quality target, parsed from the chapter's block."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    comparator: str
    threshold: float
    # Anything trailing the number in the target expression, such as
    # "for RSI on soccer_pilot_v1". Kept as written rather than parsed into
    # structure, because the scope is a note to a human reviewer and inventing a
    # schema for it would only make it look more machine-checkable than it is.
    scope: str
    frequency: str
    action_on_violation: str
    raw_target: str

    def satisfied_by(self, value: float) -> bool:
        if value != value:  # NaN: no measurement, which is not a pass
            return False
        return {
            "<": value < self.threshold,
            "<=": value <= self.threshold,
            ">": value > self.threshold,
            ">=": value >= self.threshold,
        }[self.comparator]

    def describe(self) -> str:
        return f"{self.comparator}{self.threshold:g}" + (f" {self.scope}" if self.scope else "")


class QualityFinding(BaseModel):
    """The outcome of checking one observed value against one target."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    observed: float
    target: str
    frequency: str
    passed: bool
    action: str = ""
    note: str = ""


class QualityAssessment(BaseModel):
    """Every finding from one measurement-quality review, violations first."""

    model_config = ConfigDict(extra="forbid")

    report_id: str = ""
    scenario: str = ""
    metric: str = ""
    findings: list[QualityFinding] = Field(default_factory=list)
    # Targets with no observed value. A missing measurement is not a pass, and
    # keeping it in its own list stops it being read as one.
    not_measured: list[str] = Field(default_factory=list)

    @property
    def violations(self) -> list[QualityFinding]:
        return [f for f in self.findings if not f.passed]

    @property
    def passed(self) -> bool:
        return not self.violations and not self.not_measured


def load_targets(path: str | Path = DEFAULT_TARGETS_PATH) -> dict[str, QualityTarget]:
    """Parse the ``measurement_quality_metrics`` block into checkable targets."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    block = data.get("measurement_quality_metrics")
    if not isinstance(block, dict):
        raise ValueError(f"{path}: expected a top-level measurement_quality_metrics mapping")

    targets: dict[str, QualityTarget] = {}
    for name, spec in block.items():
        raw = str(spec["target"]).strip()
        match = _TARGET_PATTERN.match(raw)
        if not match:
            raise ValueError(
                f"{name}: target {raw!r} is not a comparison this evaluator can check. "
                "Expected a form such as '<0.02' or '>0.80', optionally followed by a scope note."
            )
        targets[name] = QualityTarget(
            name=name,
            description=" ".join(str(spec.get("description", "")).split()),
            comparator=match.group("op"),
            threshold=float(match.group("value")),
            scope=match.group("scope").strip(),
            frequency=str(spec.get("frequency", "")),
            action_on_violation=" ".join(str(spec.get("action_on_violation", "")).split()),
            raw_target=raw,
        )
    return targets


def evaluate(
    observed: dict[str, float],
    targets: dict[str, QualityTarget] | None = None,
    *,
    report_id: str = "",
    scenario: str = "",
    metric: str = "",
    notes: dict[str, str] | None = None,
) -> QualityAssessment:
    """Check a measurement-quality report against the chapter's targets.

    Every target is checked, including the ones the report says nothing about.
    A target with no observed value lands in ``not_measured`` rather than
    passing quietly, because the failure mode this block exists to catch is a
    measurement nobody took.

    Values the report carries that no target covers are checked too, as long as
    a target of that name exists; anything else is ignored and left to the
    report's own consumers.
    """
    targets = targets if targets is not None else load_targets()
    notes = notes or {}
    findings: list[QualityFinding] = []
    not_measured: list[str] = []

    for name, target in targets.items():
        if name not in observed or observed[name] is None:
            not_measured.append(name)
            continue
        value = float(observed[name])
        passed = target.satisfied_by(value)
        findings.append(
            QualityFinding(
                metric=name,
                observed=value,
                target=target.describe(),
                frequency=target.frequency,
                passed=passed,
                action="" if passed else target.action_on_violation,
                note=notes.get(name, ""),
            )
        )

    findings.sort(key=lambda f: (f.passed, f.metric))
    return QualityAssessment(
        report_id=report_id,
        scenario=scenario,
        metric=metric,
        findings=findings,
        not_measured=sorted(not_measured),
    )


def load_report(path: str | Path) -> dict[str, Any]:
    """Read a measurement-quality report file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def evaluate_report(
    path: str | Path,
    targets: dict[str, QualityTarget] | None = None,
) -> QualityAssessment:
    """Evaluate a stored measurement-quality report against the targets."""
    report = load_report(path)
    observed = report.get("observed") or {}
    provenance = report.get("provenance") or {}
    return evaluate(
        {k: v for k, v in observed.items() if v is not None},
        targets,
        report_id=str(report.get("report_id", Path(path).stem)),
        scenario=str(report.get("scenario", "")),
        metric=str(report.get("metric", "")),
        notes={k: " ".join(str(v).split()) for k, v in provenance.items()},
    )


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def replicate_stability(run_values: Sequence[float]) -> float:
    """Standard deviation of a metric across independent replicate runs.

    The literal definition in the chapter's block. It needs at least two runs of
    the same scenario under unchanged conditions, and it is the measurement the
    target is written against. Where those runs do not exist,
    ``bootstrap_metric_sd`` gives a within-run approximation that will read
    lower than this one.
    """
    values = [v for v in run_values if v == v]
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1))


def bootstrap_metric_sd(
    traces: TraceSet,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> float:
    """Within-run approximation to the noise floor, by resampling prompts.

    Resamples the prompt set with replacement and recomputes the scenario value,
    returning the standard deviation across resamples. It answers a narrower
    question than the target does: how much the reading would move if a
    different sample of prompts and replicates had been drawn, holding the
    generation run itself fixed. Day-to-day variation in the model's own
    sampling is invisible to it, so the number is a floor on the noise floor and
    should be labelled as such wherever it is reported.
    """
    values = list(traces.prompt_values().values())
    if len(values) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    array = np.array(values, dtype=float)
    draws = rng.choice(array, size=(n_bootstrap, array.size), replace=True)
    return float(np.std(draws.mean(axis=1), ddof=1))


def paired_power(
    sd_of_differences: float,
    n_pairs: int,
    effect_size: float = POWER_EFFECT_SIZE,
    alpha: float = POWER_ALPHA,
) -> float:
    """Power of a paired t-test to detect ``effect_size`` at ``alpha``.

    The two-sided power of the test the aggregate layer actually runs, computed
    from the non-central t distribution rather than a normal approximation,
    because a scenario with twenty prompts has few enough degrees of freedom for
    the difference to matter.
    """
    if n_pairs < 2 or sd_of_differences != sd_of_differences or sd_of_differences <= 0:
        return float("nan")
    df = n_pairs - 1
    ncp = effect_size / (sd_of_differences / math.sqrt(n_pairs))
    crit = float(stats.t.ppf(1 - alpha / 2, df))
    return float(stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp))


def power_from_traces(
    current_traces: TraceSet,
    prior_traces: TraceSet,
    effect_size: float = POWER_EFFECT_SIZE,
    alpha: float = POWER_ALPHA,
) -> float:
    """Estimate the scenario's power from the observed spread of two runs.

    Uses the standard deviation of the per-prompt differences between the two
    runs, which is the quantity the paired test in ``compute_partitioned_delta``
    divides by. A pair of runs that happened to be quiet will overstate the
    power the scenario has in general, so this belongs in a design review
    alongside several such pairs and not in a single run's report.
    """
    current = current_traces.prompt_values()
    prior = prior_traces.prompt_values()
    shared = sorted(set(current) & set(prior))
    if len(shared) < 2:
        return float("nan")
    differences = [current[pid] - prior[pid] for pid in shared]
    return paired_power(float(np.std(differences, ddof=1)), len(differences), effect_size, alpha)


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Cohen's kappa between two raters on the same items.

    Chance-corrected agreement on nominal labels. Raw agreement would flatter
    both raters here, because the label distribution is far from uniform and two
    raters that both guessed the majority class would agree most of the time.

    Returns NaN when either rater used a single label throughout: expected
    agreement is then one, the correction divides by zero, and kappa is
    undefined rather than perfect.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("kappa needs the two raters' labels aligned on the same items")
    n = len(labels_a)
    if n == 0:
        return float("nan")
    categories = sorted(set(labels_a) | set(labels_b))
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    expected = sum(
        (labels_a.count(c) / n) * (labels_b.count(c) / n) for c in categories
    )
    if expected >= 1.0:
        return float("nan")
    return (observed - expected) / (1 - expected)


def classifier_inter_rater(
    run_a: str,
    run_b: str,
    metric: str,
    store: TraceStore | None = None,
) -> tuple[float, int]:
    """Cohen's kappa between the classifier versions used in two runs.

    Items are paired on ``output_id``, so only outputs that both runs actually
    saw enter the comparison. That pairing is the whole substance of the
    measurement: a kappa computed over two different sets of generated images
    would measure how alike the two samples were and say nothing about the two
    classifiers. In practice this means the shared evaluation set has to come
    from cached outputs, which is one more thing the caching pattern buys.

    Returns the kappa and the number of shared items behind it, because a kappa
    over a handful of items is a number with no useful precision and the count
    is the only thing that says so.
    """
    traces_a = load_traces(run_a, metric, None, store)
    traces_b = load_traces(run_b, metric, None, store)

    def _labels(traces: TraceSet) -> dict[str, str]:
        out: dict[str, str] = {}
        for record in traces.records:
            for call in record.classifier_calls:
                out[record.model_call.output_id] = call.label
        return out

    labels_a, labels_b = _labels(traces_a), _labels(traces_b)
    shared = sorted(set(labels_a) & set(labels_b))
    if not shared:
        return float("nan"), 0
    return cohen_kappa([labels_a[k] for k in shared], [labels_b[k] for k in shared]), len(shared)


def measure_from_traces(
    current_run_id: str,
    prior_run_id: str,
    metric: str,
    store: TraceStore | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Produce a measurement-quality report for a pair of runs, with provenance.

    Returns the observed values and, alongside each, a plain note saying how it
    was obtained. The notes are not decoration: two of these three numbers are
    approximations to what the target is written against, and a report that
    carried the values without the caveats would invite a reader to treat an
    optimistic estimate as the measurement.
    """
    current = load_traces(current_run_id, metric, None, store)
    prior = load_traces(prior_run_id, metric, None, store)

    stability = bootstrap_metric_sd(current)
    power = power_from_traces(current, prior)
    kappa, n_shared = classifier_inter_rater(prior_run_id, current_run_id, metric, store)

    observed = {
        "replicate_stability": stability,
        "sample_size_power": power,
    }
    provenance = {
        "replicate_stability": (
            f"Bootstrap over the {len(current.prompt_values())} prompts of {current_run_id}. "
            "A within-run approximation and a lower bound on the true noise floor, which "
            "needs independent replicate runs to measure."
        ),
        "sample_size_power": (
            f"Paired power to detect a {POWER_EFFECT_SIZE} effect at alpha={POWER_ALPHA}, from the "
            "observed spread of per-prompt differences between "
            f"{prior_run_id} and {current_run_id}. "
            "Estimated after the fact, so it describes this pair rather than the scenario."
        ),
    }

    if kappa == kappa and n_shared > 0:
        observed["classifier_inter_rater"] = kappa
        provenance["classifier_inter_rater"] = (
            f"Cohen's kappa between the classifier versions of {prior_run_id} and "
            f"{current_run_id} over {n_shared} shared cached outputs."
        )
    else:
        provenance["classifier_inter_rater"] = (
            f"Not computed: {prior_run_id} and {current_run_id} share no cached outputs, so "
            "there is no shared evaluation set to compare the classifier versions on."
        )
    return observed, provenance


def summarize(assessment: QualityAssessment, *, width: int = 78) -> str:
    """Render an assessment as plain text for a terminal or an incident note."""
    lines: list[str] = []
    header = "MEASUREMENT QUALITY"
    if assessment.report_id:
        header += f"  {assessment.report_id}"
    lines.append(header)
    context = " / ".join(x for x in (assessment.scenario, assessment.metric) if x)
    if context:
        lines.append(context)
    lines.append("-" * width)
    for finding in assessment.findings:
        status = "PASS" if finding.passed else "FAIL"
        measured = finding.observed == finding.observed
        value = f"{finding.observed:.4f}" if measured else "no measurement"
        lines.append(
            f"{status}  {finding.metric:<24} observed {value:>14}  target {finding.target}"
        )
        lines.append(f"      checked {finding.frequency}")
        if finding.note:
            lines.append(_wrap(f"provenance: {finding.note}", width, "      "))
        if not finding.passed:
            lines.append(_wrap(f"action: {finding.action}", width, "      "))
    for name in assessment.not_measured:
        lines.append(f"----  {name:<24} not measured in this report")
    lines.append("-" * width)
    if assessment.passed:
        lines.append("All measurement-quality targets met.")
    else:
        breached = ", ".join(f.metric for f in assessment.violations)
        headline = f"{len(assessment.violations)} target(s) breached"
        parts = [headline + (f": {breached}" if breached else "")]
        if assessment.not_measured:
            parts.append(f"{len(assessment.not_measured)} not measured")
        lines.append("; ".join(parts) + ".")
    return "\n".join(lines)


def _wrap(text: str, width: int, indent: str) -> str:
    import textwrap

    return "\n".join(
        textwrap.wrap(text, width=width, initial_indent=indent, subsequent_indent=indent + "  ")
    )


def _iter_targets(targets: Iterable[QualityTarget]) -> list[str]:
    return [t.name for t in targets]
