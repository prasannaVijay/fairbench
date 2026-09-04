"""The three-layer fairness-regression investigation, packaged as one call.

Book: Chapter 10, "A three-layer debugging model" and "Traces as the debugging
backbone".

The chapter prints ``investigate_regression`` and describes the shape its result
takes: ``partitioned_delta`` carries the aggregate check, ``distribution_shift``
the distributional one, ``top_contributing_prompts`` the instance layer, and
``classifier_confidence_shift`` is the guard that separates a real model
regression from a classifier that simply changed its mind. This module is that
function and the four helpers behind it.

What the report can and cannot support is worth stating before the code. It is
an ordered set of observations, and the ordering is deliberate: the aggregate
layer establishes that something moved and where, the distributional layer asks
whether the model's own output distribution moved with it, the classifier guard
asks whether the instrument moved instead, and the instance layer names the
prompts that carried the movement so a human can look at them. The report does
not establish causation. When both the output distribution and the classifier
confidence distribution have shifted, the two hypotheses are confounded and the
report says so rather than picking one; the honest next step in that case is a
re-score of the current run's cached outputs under the prior classifier version,
which the report recommends but cannot perform on its own.

The statistical tests here are screening tests, chosen because they are cheap
and make few assumptions, and they are run on data that was not collected under
a pre-registered design. Their p-values are best read as a ranking of which
signals deserve a human's attention, and treating them as confirmatory evidence
would overstate what a single pair of benchmark runs can tell us.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats
from trace_store import ClassifierCall, ModelCall, TraceSet, load_traces

# Screening thresholds. They are conventions, not findings, and a team should
# calibrate them against its own noise floor before wiring them to an alert.
# The 0.02 on the aggregate delta is the replicate-stability target from the
# chapter's measurement-quality block: a movement smaller than the metric's own
# noise floor is not a movement worth investigating.
NOISE_FLOOR = 0.02
ALPHA = 0.05
# Jensen-Shannon distance above which a categorical distribution is treated as
# having moved enough to matter, alongside the significance test.
JS_DISTANCE_THRESHOLD = 0.05
MAX_CONTRIBUTING_PROMPTS = 20


class Verdict(str, Enum):
    """What the four signals, read together, support.

    Deliberately coarse. This is about as much resolution as a single pair of
    benchmark runs will carry, and a finer scale would invite readers to treat
    the label as a diagnosis when it is a routing decision. Two of the five
    values, ``confounded`` and ``unexplained``, exist so that the report can
    decline to choose when the evidence does not separate the hypotheses.
    """

    NO_MOVEMENT = "no_movement"                    # inside the noise floor
    MODEL_REGRESSION_SUSPECTED = "model_regression_suspected"
    CLASSIFIER_SHIFT_SUSPECTED = "classifier_shift_suspected"
    CONFOUNDED = "confounded"                      # both moved; cannot separate
    UNEXPLAINED = "unexplained"                    # metric moved, neither did


# ---------------------------------------------------------------------------
# Layer 1: aggregate partition analysis
# ---------------------------------------------------------------------------


class PartitionDelta(BaseModel):
    """The movement within one slice of the scenario."""

    model_config = ConfigDict(extra="forbid")

    partition: str
    # The slice's metric value, a mean across the prompts in it, matching the
    # aggregation the scenario value uses.
    current_value: float
    prior_value: float
    delta: float
    # Scored trace records behind the slice, not prompts. Kept as records
    # because this is the number that says how thin a slice has become, and a
    # slice whose prompts survived while most of their replicates did not is
    # exactly the case worth noticing.
    n_current: int
    n_prior: int
    # Share of the scenario-level movement this slice accounts for, as a signed
    # fraction. Slices that moved against the regression get a negative share.
    share_of_movement: float


class PartitionedDelta(BaseModel):
    """Layer 1. Whether the regression is real, and whether it is localized.

    The scenario-level movement first, then the same movement decomposed across
    every dimension the traces were partitioned along. A movement spread evenly
    across all slices points upstream of any one scenario; a movement carried by
    two clusters out of four points at something those two clusters share.
    """

    model_config = ConfigDict(extra="forbid")

    metric: str
    current_run_id: str
    prior_run_id: str
    attribute_partition: str | None
    current_value: float
    prior_value: float
    delta: float
    n_current: int
    n_prior: int
    # Welch's t-test on the per-record contributions. Unequal variances are the
    # norm across runs, so the pooled-variance form would understate the test's
    # own uncertainty.
    p_value: float
    # True when the two runs shared a prompt library and the test could be
    # paired. Between-prompt heterogeneity is large in a fairness scenario, and
    # pairing removes it from the comparison entirely.
    paired: bool
    # A 95% interval on the difference in means, reported because a reader
    # comparing two runs needs the width as much as the point estimate.
    delta_confidence_interval: tuple[float, float]
    exceeds_noise_floor: bool
    significant: bool
    # Exclusion rates are reported alongside the delta rather than folded into
    # it. A rise here is the chapter's sample-attrition hazard, and it is worth
    # an alert of its own even in a run where the fairness metric held steady.
    current_exclusion_rate: float
    prior_exclusion_rate: float
    unmeasured_prompts_current: list[str] = Field(default_factory=list)
    unmeasured_prompts_prior: list[str] = Field(default_factory=list)
    by_cluster: list[PartitionDelta] = Field(default_factory=list)
    by_attribute: dict[str, list[PartitionDelta]] = Field(default_factory=dict)

    @property
    def localized_clusters(self) -> list[str]:
        """Clusters carrying more than a proportional share of the movement."""
        if not self.by_cluster:
            return []
        proportional = 1.0 / len(self.by_cluster)
        return [p.partition for p in self.by_cluster if p.share_of_movement > proportional]


# ---------------------------------------------------------------------------
# Layer 2a: distributional shift in the model's own outputs
# ---------------------------------------------------------------------------


class DistributionShift(BaseModel):
    """Layer 2. Whether the model's output distribution itself moved.

    Read off the pinned reference probe rather than the production classifier,
    so that this signal stays still when the classifier is upgraded. If the
    probe were upgraded in step with the classifier, this field and the
    confidence field would move together and the guard below would be useless.
    """

    model_config = ConfigDict(extra="forbid")

    reference_probe: str
    current_distribution: dict[str, float]
    prior_distribution: dict[str, float]
    category_deltas: dict[str, float]
    # Jensen-Shannon distance: symmetric, bounded in [0, 1], and defined when a
    # category appears in one run and not the other, which KL divergence is not.
    js_distance: float
    chi2_p_value: float
    n_current: int
    n_prior: int
    shifted: bool


# ---------------------------------------------------------------------------
# Layer 2b: the classifier-confidence guard
# ---------------------------------------------------------------------------


class ConfidenceShift(BaseModel):
    """The guard that separates a real model regression from a changed instrument.

    A classifier that has become less certain is a classifier that is excluding
    more outputs from scoring, and outputs excluded unevenly across subgroups
    move a fairness metric without the model having generated anything
    different. Three readings carry that: the central tendency of the confidence
    distribution, the fraction of judgements that fell below the scoring
    threshold, and whether the classifier version changed between the runs at
    all. The last one is the cheapest and, in practice, often the decisive one.
    """

    model_config = ConfigDict(extra="forbid")

    current_versions: list[str]
    prior_versions: list[str]
    version_changed: bool
    current_mean_confidence: float
    prior_mean_confidence: float
    mean_delta: float
    current_median_confidence: float
    prior_median_confidence: float
    current_borderline_rate: float
    prior_borderline_rate: float
    borderline_rate_delta: float
    # Two-sample Kolmogorov-Smirnov on the raw confidence values. Chosen over a
    # test of means because the failure mode the chapter describes is a thicker
    # tail of borderline judgements, which can leave the mean almost untouched.
    ks_statistic: float
    ks_p_value: float
    n_current: int
    n_prior: int
    shifted: bool


# ---------------------------------------------------------------------------
# Layer 3: instance-level contribution
# ---------------------------------------------------------------------------


class PromptContribution(BaseModel):
    """One prompt's share of the aggregate movement.

    ``contribution`` is signed towards the observed move: a prompt that pushed
    the metric in the direction the scenario moved gets a positive number, and a
    prompt that pushed against it gets a negative one. The contributions sum to
    the absolute value of the scenario delta, so the ranking is a decomposition
    and not a heuristic score.
    """

    model_config = ConfigDict(extra="forbid")

    prompt_id: str
    prompt: str
    prompt_cluster: str
    attributes: dict[str, str] = Field(default_factory=dict)
    current_value: float
    prior_value: float
    delta: float
    contribution: float
    share_of_movement: float
    # Carried alongside so a reader ranking by contribution can see, on the same
    # row, whether the classifier also became less certain about this prompt.
    confidence_delta: float
    borderline_rate_delta: float
    excluded_current: int
    excluded_prior: int
    n_current: int
    n_prior: int


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


class RegressionReport(BaseModel):
    """The structured artifact a single investigation produces.

    Its five fields map onto the chapter's three layers plus the guard:

    - ``partitioned_delta`` is the aggregate layer: did the metric really move,
      and is the movement localized to particular slices?
    - ``distribution_shift`` is the distributional layer: did the model's own
      output distribution move, independently of how it was classified?
    - ``classifier_confidence_shift`` is the guard: did the instrument move
      instead? This is the field that separates a real model regression from a
      classifier that simply changed its mind.
    - ``top_contributing_prompts`` is the instance layer: which prompts carried
      the movement, so that a human can read their traces?

    ``metric`` names what all four are about. The run identifiers and partition
    default to empty so the constructor the chapter prints, which passes the
    five substantive fields, works unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    metric: str
    partitioned_delta: PartitionedDelta
    distribution_shift: DistributionShift
    classifier_confidence_shift: ConfidenceShift
    # Capped at twenty. The cap is a reading-time constraint rather than a
    # statistical one: a list long enough that nobody opens the traces has
    # stopped being an instance-layer investigation.
    top_contributing_prompts: list[PromptContribution] = Field(
        default_factory=list, max_length=MAX_CONTRIBUTING_PROMPTS
    )

    # The run identifiers and the partition are read back off the aggregate
    # layer rather than stored twice, which keeps the constructor to the five
    # fields the chapter prints and removes any chance of the two disagreeing.
    @property
    def current_run_id(self) -> str:
        return self.partitioned_delta.current_run_id

    @property
    def prior_run_id(self) -> str:
        return self.partitioned_delta.prior_run_id

    @property
    def attribute_partition(self) -> str | None:
        return self.partitioned_delta.attribute_partition

    @property
    def verdict(self) -> Verdict:
        """Which hypothesis the four signals support, or that they do not separate.

        This is a routing decision and not a diagnosis. It says which of the
        chapter's root-cause categories to open first, and the confounded case
        exists because two runs sometimes cannot tell us.
        """
        moved = self.partitioned_delta.exceeds_noise_floor
        model_moved = self.distribution_shift.shifted
        instrument_moved = self.classifier_confidence_shift.shifted
        if not moved:
            return Verdict.NO_MOVEMENT
        if model_moved and instrument_moved:
            return Verdict.CONFOUNDED
        if model_moved:
            return Verdict.MODEL_REGRESSION_SUSPECTED
        if instrument_moved:
            return Verdict.CLASSIFIER_SHIFT_SUSPECTED
        return Verdict.UNEXPLAINED

    @property
    def recommended_next_step(self) -> str:
        """What to do next, phrased so that it can be pasted into an incident note."""
        return {
            Verdict.NO_MOVEMENT: (
                "Movement is inside the metric's noise floor. Record the reading and "
                "leave the alert threshold alone unless it fires again next run."
            ),
            Verdict.MODEL_REGRESSION_SUSPECTED: (
                "Output distribution moved while the classifier held steady. Open the "
                "model change log for the version boundary and read the traces of the "
                "top contributing prompts before escalating."
            ),
            Verdict.CLASSIFIER_SHIFT_SUSPECTED: (
                "Classifier confidence moved while the output distribution held steady. "
                "Re-score the current run's cached outputs under the prior classifier "
                "version; if the metric returns to its prior value, pin the classifier "
                "version for the comparison and treat the reading as an instrument "
                "change rather than a model regression."
            ),
            Verdict.CONFOUNDED: (
                "Both the output distribution and the classifier confidence moved, so "
                "this pair of runs cannot separate them. Re-score the current outputs "
                "under the prior classifier version to hold the instrument fixed, and "
                "only then compare."
            ),
            Verdict.UNEXPLAINED: (
                "The metric moved without a detectable shift in either the outputs or "
                "the classifier. Check the events timeline for prompt-library "
                "revisions and reference-distribution updates before assuming noise."
            ),
        }[self.verdict]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _distribution(labels: list[str]) -> dict[str, float]:
    if not labels:
        return {}
    total = len(labels)
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return {k: counts[k] / total for k in sorted(counts)}


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    divergence = 0.5 * stats.entropy(p, m, base=2) + 0.5 * stats.entropy(q, m, base=2)
    return float(math.sqrt(max(divergence, 0.0)))


def compute_partitioned_delta(
    current_traces: TraceSet,
    prior_traces: TraceSet,
) -> PartitionedDelta:
    """Layer 1. Establish that the metric moved, and locate where it moved.

    The scenario delta comes first, with a significance test and a confidence
    interval on the difference, so that the width of the estimate travels
    alongside the point. The decomposition follows: per prompt cluster, and per
    value of every declared prompt attribute the traces carry. A movement spread
    evenly across all slices is a different problem from one carried by two, and
    separating those two readings is the first thing an on-call engineer needs.

    The comparison runs on per-prompt values rather than per-record ones, and
    the test is paired whenever the two runs share their prompt library. Prompts
    in a fairness scenario differ enormously from each other by construction,
    since that variety is the point of the library, and an unpaired test spends
    all of its power on that between-prompt spread before it reaches the
    run-over-run difference we care about. When the libraries differ, which a
    prompt-library revision will cause and which is one of the chapter's own
    root-cause categories, the comparison falls back to Welch's t-test on the
    two sets of prompt values and reports ``paired=False`` so the reader knows
    the test lost power.

    ``share_of_movement`` is normalized by the scenario delta, so the shares sum
    to one and a slice that moved against the regression reads as negative.
    """
    current_values = current_traces.prompt_values()
    prior_values = prior_traces.prompt_values()
    current_value = _mean(list(current_values.values()))
    prior_value = _mean(list(prior_values.values()))
    delta = current_value - prior_value

    shared = sorted(set(current_values) & set(prior_values))
    paired = len(shared) >= 3 and set(current_values) == set(prior_values)
    if paired:
        differences = [current_values[pid] - prior_values[pid] for pid in shared]
        result = stats.ttest_rel(
            [current_values[pid] for pid in shared],
            [prior_values[pid] for pid in shared],
        )
        p_value = float(result.pvalue)
        se = float(np.std(differences, ddof=1)) / math.sqrt(len(differences))
        crit = float(stats.t.ppf(1 - ALPHA / 2, len(differences) - 1))
        ci = (float(np.mean(differences)) - crit * se, float(np.mean(differences)) + crit * se)
    elif len(current_values) > 1 and len(prior_values) > 1:
        cur_list = list(current_values.values())
        pri_list = list(prior_values.values())
        p_value = float(stats.ttest_ind(cur_list, pri_list, equal_var=False).pvalue)
        se = math.sqrt(
            float(np.var(cur_list, ddof=1)) / len(cur_list)
            + float(np.var(pri_list, ddof=1)) / len(pri_list)
        )
        df = _welch_df(cur_list, pri_list)
        crit = float(stats.t.ppf(1 - ALPHA / 2, df)) if df > 0 else float("nan")
        ci = (delta - crit * se, delta + crit * se)
    else:
        p_value = float("nan")
        ci = (float("nan"), float("nan"))

    def _slice_values(records: list) -> dict[str, float]:
        totals: dict[str, list[float]] = {}
        for r in records:
            totals.setdefault(r.prompt_id, []).append(r.metric_contribution)
        return {pid: sum(v) / len(v) for pid, v in totals.items()}

    n_prompts_current = len(current_values)
    n_prompts_prior = len(prior_values)

    def _slices(
        current_groups: dict[str, list],
        prior_groups: dict[str, list],
    ) -> list[PartitionDelta]:
        out: list[PartitionDelta] = []
        for name in sorted(set(current_groups) | set(prior_groups)):
            cur = _slice_values(current_groups.get(name, []))
            pri = _slice_values(prior_groups.get(name, []))
            cur_v, pri_v = _mean(list(cur.values())), _mean(list(pri.values()))
            # The slice's additive share of the scenario mean, on each side. The
            # difference of the two shares is exactly what this slice added to
            # the scenario movement.
            cur_share = sum(cur.values()) / n_prompts_current if n_prompts_current else 0.0
            pri_share = sum(pri.values()) / n_prompts_prior if n_prompts_prior else 0.0
            share = ((cur_share - pri_share) / delta) if delta else 0.0
            out.append(
                PartitionDelta(
                    partition=name,
                    current_value=cur_v,
                    prior_value=pri_v,
                    delta=cur_v - pri_v,
                    n_current=len(current_groups.get(name, [])),
                    n_prior=len(prior_groups.get(name, [])),
                    share_of_movement=share,
                )
            )
        return sorted(out, key=lambda p: p.share_of_movement, reverse=True)

    attribute_keys = sorted(
        {k for r in current_traces.scored_records for k in r.attributes}
        | {k for r in prior_traces.scored_records for k in r.attributes}
    )
    by_attribute = {
        key: _slices(current_traces.by_attribute(key), prior_traces.by_attribute(key))
        for key in attribute_keys
    }

    return PartitionedDelta(
        metric=current_traces.metric,
        current_run_id=current_traces.run_id,
        prior_run_id=prior_traces.run_id,
        attribute_partition=current_traces.attribute_partition,
        current_value=current_value,
        prior_value=prior_value,
        delta=delta,
        n_current=len(current_traces.scored_records),
        n_prior=len(prior_traces.scored_records),
        p_value=p_value,
        paired=paired,
        delta_confidence_interval=ci,
        exceeds_noise_floor=abs(delta) > NOISE_FLOOR,
        significant=bool(p_value == p_value and p_value < ALPHA),
        current_exclusion_rate=current_traces.exclusion_rate,
        prior_exclusion_rate=prior_traces.exclusion_rate,
        unmeasured_prompts_current=current_traces.unmeasured_prompts,
        unmeasured_prompts_prior=prior_traces.unmeasured_prompts,
        by_cluster=_slices(current_traces.by_cluster(), prior_traces.by_cluster()),
        by_attribute=by_attribute,
    )


def _welch_df(a: list[float], b: list[float]) -> float:
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    na, nb = len(a), len(b)
    numerator = (va / na + vb / nb) ** 2
    denominator = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    return numerator / denominator if denominator > 0 else float(na + nb - 2)


def compare_output_distributions(
    current_outputs: list[ModelCall],
    prior_outputs: list[ModelCall],
) -> DistributionShift:
    """Layer 2. Ask whether the model's own output distribution moved.

    The comparison runs on the pinned reference probe's labels, which is the
    whole point: the probe is held at one version across runs so that this
    signal is independent of the production classifier. A chi-square test of
    independence on the raw counts answers whether the two runs look like draws
    from the same distribution, and the Jensen-Shannon distance says how far
    apart they are, because a large sample will return a significant p-value for
    a difference too small to explain a metric movement.

    Both readings have to clear their threshold before ``shifted`` is set. A
    significant test on a distance of 0.01 is a true statement about sampling
    and a misleading one about the model.
    """
    probe_names = sorted({m.reference_probe for m in current_outputs + prior_outputs})
    current_labels = [m.reference_probe_label for m in current_outputs]
    prior_labels = [m.reference_probe_label for m in prior_outputs]
    current_dist = _distribution(current_labels)
    prior_dist = _distribution(prior_labels)
    categories = sorted(set(current_dist) | set(prior_dist))
    deltas = {
        c: current_dist.get(c, 0.0) - prior_dist.get(c, 0.0) for c in categories
    }

    table = np.array(
        [
            [current_labels.count(c) for c in categories],
            [prior_labels.count(c) for c in categories],
        ],
        dtype=float,
    )
    # Drop categories absent from both runs; chi-square is undefined on an
    # all-zero column and would otherwise return nan for the whole table.
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] >= 2 and table.sum() > 0:
        chi2_p = float(stats.chi2_contingency(table).pvalue)
    else:
        chi2_p = float("nan")

    js = _jsd_from_dicts(current_dist, prior_dist)
    shifted = bool(js > JS_DISTANCE_THRESHOLD and chi2_p == chi2_p and chi2_p < ALPHA)

    return DistributionShift(
        reference_probe="+".join(probe_names) if probe_names else "unknown",
        current_distribution=current_dist,
        prior_distribution=prior_dist,
        category_deltas=deltas,
        js_distance=js,
        chi2_p_value=chi2_p,
        n_current=len(current_labels),
        n_prior=len(prior_labels),
        shifted=shifted,
    )


def _jsd_from_dicts(p: dict[str, float], q: dict[str, float]) -> float:
    keys = sorted(set(p) | set(q))
    if not keys:
        return 0.0
    pv = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    qv = np.array([q.get(k, 0.0) for k in keys], dtype=float)
    if pv.sum() <= 0 or qv.sum() <= 0:
        return float("nan")
    return _jsd(pv / pv.sum(), qv / qv.sum())


def compare_confidence_distributions(
    current_classifications: list[ClassifierCall],
    prior_classifications: list[ClassifierCall],
) -> ConfidenceShift:
    """The guard. Ask whether the instrument moved instead of the model.

    Three readings, in increasing order of how much work they take to interpret.
    A classifier version change between the runs is a fact, not an inference,
    and on its own it is enough to hold a cross-boundary comparison for review.
    The borderline rate is next: a rise means more outputs dropped out of
    scoring, and outputs that drop out unevenly across subgroups move a fairness
    metric with no change in what the model generated. The Kolmogorov-Smirnov
    test comes last and is the most sensitive of the three, because the failure
    mode the chapter describes, a thicker tail of uncertain judgements on one
    kind of image, can leave the mean confidence almost where it was.

    ``shifted`` is set when the distribution test clears significance and either
    the version changed or the borderline rate moved by more than two points.
    Requiring the second condition keeps a very large sample from flagging a
    statistically detectable and operationally irrelevant drift.
    """
    current_scores = [c.confidence for c in current_classifications]
    prior_scores = [c.confidence for c in prior_classifications]
    current_versions = sorted(
        {f"{c.classifier}@{c.classifier_version}" for c in current_classifications}
    )
    prior_versions = sorted(
        {f"{c.classifier}@{c.classifier_version}" for c in prior_classifications}
    )

    if current_scores and prior_scores:
        ks = stats.ks_2samp(current_scores, prior_scores)
        ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
    else:
        ks_stat, ks_p = float("nan"), float("nan")

    current_borderline = (
        sum(1 for c in current_classifications if c.borderline) / len(current_classifications)
        if current_classifications
        else float("nan")
    )
    prior_borderline = (
        sum(1 for c in prior_classifications if c.borderline) / len(prior_classifications)
        if prior_classifications
        else float("nan")
    )
    borderline_delta = current_borderline - prior_borderline
    version_changed = current_versions != prior_versions

    material = version_changed or abs(borderline_delta) > 0.02
    shifted = bool(ks_p == ks_p and ks_p < ALPHA and material)

    return ConfidenceShift(
        current_versions=current_versions,
        prior_versions=prior_versions,
        version_changed=version_changed,
        current_mean_confidence=_mean(current_scores),
        prior_mean_confidence=_mean(prior_scores),
        mean_delta=_mean(current_scores) - _mean(prior_scores),
        current_median_confidence=(
            float(np.median(current_scores)) if current_scores else float("nan")
        ),
        prior_median_confidence=(
            float(np.median(prior_scores)) if prior_scores else float("nan")
        ),
        current_borderline_rate=current_borderline,
        prior_borderline_rate=prior_borderline,
        borderline_rate_delta=borderline_delta,
        ks_statistic=ks_stat,
        ks_p_value=ks_p,
        n_current=len(current_scores),
        n_prior=len(prior_scores),
        shifted=shifted,
    )


def rank_prompts_by_contribution(
    current_traces: TraceSet,
    prior_traces: TraceSet,
    metric: str,
) -> list[PromptContribution]:
    """Layer 3. Return the prompts ordered by how much each moved the metric.

    Because the scenario value is the mean across prompts, each prompt's share
    of that mean is simply its value divided by the number of prompts, and the
    movement it contributed is the difference of its shares between the two
    runs. Those contributions sum exactly to the scenario delta, so the list is
    a decomposition of the movement and not a heuristic score. That exactness is
    what stops a reader over-attributing the top row.

    The ordering is descending in the direction the scenario moved, so the
    prompts that carried the regression come first and prompts that pushed the
    other way sit at the bottom with negative contributions. The bottom rows are
    worth reading too: a movement that is large at the top and large-negative at
    the bottom is a redistribution inside the scenario, which is a different
    finding from a uniform drift.

    Each row carries the prompt's classifier-confidence change and its exclusion
    counts alongside its contribution, so that the instance layer and the guard
    can be read on one line instead of in two queries. A prompt that rose while
    its classifier confidence fell and its exclusions climbed is the signature
    the chapter's worked example turns on.
    """
    current_values = current_traces.prompt_values()
    prior_values = prior_traces.prompt_values()
    n_current = len(current_values)
    n_prior = len(prior_values)

    scenario_delta = _mean(list(current_values.values())) - _mean(list(prior_values.values()))
    direction = 1.0 if scenario_delta >= 0 else -1.0

    current_records: dict[str, list] = {}
    prior_records: dict[str, list] = {}
    for r in current_traces.records:
        current_records.setdefault(r.prompt_id, []).append(r)
    for r in prior_traces.records:
        prior_records.setdefault(r.prompt_id, []).append(r)

    def _confidence(records: list) -> tuple[float, float]:
        calls = [c for r in records for c in r.classifier_calls]
        if not calls:
            return float("nan"), float("nan")
        mean_conf = sum(c.confidence for c in calls) / len(calls)
        borderline = sum(1 for c in calls if c.borderline) / len(calls)
        return mean_conf, borderline

    rows: list[PromptContribution] = []
    for prompt_id in sorted(set(current_records) | set(prior_records)):
        cur_records = current_records.get(prompt_id, [])
        pri_records = prior_records.get(prompt_id, [])
        # All records for a prompt carry the same text and cluster; take the
        # current run's copy where it exists, so a revised prompt reads as it is
        # now rather than as it was.
        sample = (cur_records or pri_records)[0]
        cur_v = current_values.get(prompt_id)
        pri_v = prior_values.get(prompt_id)
        # A prompt that was measured in only one of the two runs still moved the
        # scenario mean, by entering or leaving it. Its share on the missing side
        # is zero, which is the arithmetic that makes the decomposition exact.
        cur_share = (cur_v / n_current) if (cur_v is not None and n_current) else 0.0
        pri_share = (pri_v / n_prior) if (pri_v is not None and n_prior) else 0.0
        contribution = (cur_share - pri_share) * direction
        cur_conf, cur_border = _confidence(cur_records)
        pri_conf, pri_border = _confidence(pri_records)
        rows.append(
            PromptContribution(
                prompt_id=prompt_id,
                prompt=sample.prompt,
                prompt_cluster=sample.prompt_cluster,
                attributes=dict(sample.attributes),
                current_value=cur_v if cur_v is not None else float("nan"),
                prior_value=pri_v if pri_v is not None else float("nan"),
                delta=(
                    (cur_v - pri_v)
                    if (cur_v is not None and pri_v is not None)
                    else float("nan")
                ),
                contribution=contribution,
                share_of_movement=(contribution / abs(scenario_delta)) if scenario_delta else 0.0,
                confidence_delta=cur_conf - pri_conf,
                borderline_rate_delta=cur_border - pri_border,
                excluded_current=sum(1 for r in cur_records if not r.scored),
                excluded_prior=sum(1 for r in pri_records if not r.scored),
                n_current=sum(1 for r in cur_records if r.scored),
                n_prior=sum(1 for r in pri_records if r.scored),
            )
        )

    rows.sort(key=lambda row: row.contribution, reverse=True)
    return rows


def investigate_regression(
    metric: str,
    current_run_id: str,
    prior_run_id: str,
    attribute_partition: str | None = None,
) -> RegressionReport:
    """Walk the aggregate, distributional, and instance layers in one call.

    This is the function the chapter prints. The alternative most teams end up
    with is a collection of ad-hoc queries and notebook cells that produce the
    same information more slowly and in a form the next investigation cannot
    reuse; packaging the walk means the analysis is already waiting alongside
    the anomaly when someone comes to look at it.

    Args:
        metric: The fairness metric that moved, e.g. ``"RSI_gender"``.
        current_run_id: The run the alert fired on.
        prior_run_id: The run to compare it against, normally the last one that
            was accepted as a clean reading.
        attribute_partition: An optional slice to restrict the investigation to,
            given as a conjunction of ``key=value`` clauses over the prompt's
            declared attributes.

    Returns:
        A ``RegressionReport``. Read ``verdict`` for the routing decision and
        the four fields for the evidence behind it.
    """
    current_traces = load_traces(current_run_id, metric, attribute_partition)
    prior_traces = load_traces(prior_run_id, metric, attribute_partition)

    # Layer 1: aggregate partition analysis
    partitioned_delta = compute_partitioned_delta(current_traces, prior_traces)

    # Layer 2: distributional shift analysis
    distribution_shift = compare_output_distributions(
        current_traces.outputs, prior_traces.outputs
    )
    classifier_confidence_shift = compare_confidence_distributions(
        current_traces.classifications, prior_traces.classifications
    )

    # Layer 3: instance-level contribution
    driving_prompts = rank_prompts_by_contribution(
        current_traces, prior_traces, metric
    )

    return RegressionReport(
        metric=metric,
        partitioned_delta=partitioned_delta,
        distribution_shift=distribution_shift,
        classifier_confidence_shift=classifier_confidence_shift,
        top_contributing_prompts=driving_prompts[:20],
    )
