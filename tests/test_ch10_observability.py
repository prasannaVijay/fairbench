"""Tests for the Chapter 10 observability code.

The behaviours worth pinning down are the ones the chapter's argument rests on:
that a single call produces all three layers plus the guard, that the instance
layer is an ordered decomposition rather than a heuristic ranking, that the
guard actually separates a classifier change from a model regression, and that
the measurement-quality targets fail loudly when they are breached.
"""

import sys
from pathlib import Path

import pytest

_CH10 = Path(__file__).resolve().parent.parent / "ch10"
sys.path.insert(0, str(_CH10))

from measurement_quality import (  # noqa: E402
    classifier_inter_rater,
    cohen_kappa,
    evaluate,
    evaluate_report,
    load_targets,
    paired_power,
    replicate_stability,
)
from regression_report import (  # noqa: E402
    MAX_CONTRIBUTING_PROMPTS,
    RegressionReport,
    Verdict,
    compare_confidence_distributions,
    investigate_regression,
    rank_prompts_by_contribution,
)
from trace_store import (  # noqa: E402
    ClassifierCall,
    ModelCall,
    TraceRecord,
    TraceStore,
    load_traces,
    parse_partition,
)

CLASSIFIER_SHIFT = ("soccer_2026_w14", "soccer_2026_w13")
MODEL_REGRESSION = ("soccer_2026_w21", "soccer_2026_w20")
FIXTURES = _CH10 / "fixtures"


# ---------------------------------------------------------------------------
# The report as a whole
# ---------------------------------------------------------------------------


def test_investigate_regression_populates_all_five_fields() -> None:
    current, prior = CLASSIFIER_SHIFT
    report = investigate_regression("RSI_gender", current, prior)

    assert isinstance(report, RegressionReport)
    assert report.metric == "RSI_gender"
    # Layer 1
    assert report.partitioned_delta.n_current > 0
    assert report.partitioned_delta.by_cluster
    assert report.partitioned_delta.delta != 0
    # Layer 2
    assert report.distribution_shift.current_distribution
    assert report.distribution_shift.prior_distribution
    # The guard
    assert report.classifier_confidence_shift.current_versions
    assert report.classifier_confidence_shift.n_current > 0
    # Layer 3
    assert report.top_contributing_prompts
    assert all(p.prompt_id and p.prompt for p in report.top_contributing_prompts)
    # And the run identifiers read back off the aggregate layer.
    assert report.current_run_id == current
    assert report.prior_run_id == prior


def test_default_partition_is_none_and_signature_matches_the_book() -> None:
    import inspect

    signature = inspect.signature(investigate_regression)
    assert list(signature.parameters) == [
        "metric",
        "current_run_id",
        "prior_run_id",
        "attribute_partition",
    ]
    assert signature.parameters["attribute_partition"].default is None


# ---------------------------------------------------------------------------
# Layer 3: the ordering and the cap
# ---------------------------------------------------------------------------


def test_top_contributing_prompts_are_ordered_by_contribution() -> None:
    current, prior = CLASSIFIER_SHIFT
    report = investigate_regression("RSI_gender", current, prior)
    contributions = [p.contribution for p in report.top_contributing_prompts]
    assert contributions == sorted(contributions, reverse=True)


def test_contributions_decompose_the_aggregate_movement() -> None:
    """Every prompt's contribution sums back to the scenario delta.

    This is the property that makes the ranking a decomposition. Without it a
    reader could not tell whether the top row explains a third of the movement
    or a twentieth.
    """
    current, prior = CLASSIFIER_SHIFT
    rows = rank_prompts_by_contribution(
        load_traces(current, "RSI_gender"),
        load_traces(prior, "RSI_gender"),
        "RSI_gender",
    )
    report = investigate_regression("RSI_gender", current, prior)
    assert sum(r.contribution for r in rows) == pytest.approx(
        abs(report.partitioned_delta.delta), abs=1e-9
    )


def test_top_contributing_prompts_are_capped_at_twenty(tmp_path) -> None:
    """A scenario with more than twenty prompts still reports twenty rows."""
    store = _synthetic_store(tmp_path, n_prompts=26)
    rows = rank_prompts_by_contribution(
        load_traces("run_b", "RSI_gender", None, store),
        load_traces("run_a", "RSI_gender", None, store),
        "RSI_gender",
    )
    assert len(rows) == 26
    assert len(rows[:MAX_CONTRIBUTING_PROMPTS]) == 20


def test_report_rejects_more_than_twenty_contributing_prompts() -> None:
    current, prior = CLASSIFIER_SHIFT
    report = investigate_regression("RSI_gender", current, prior)
    payload = report.model_dump()
    payload["top_contributing_prompts"] = payload["top_contributing_prompts"] * 3
    with pytest.raises(Exception):
        RegressionReport.model_validate(payload)


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_classifier_shift_is_distinguished_from_model_regression() -> None:
    """The two investigations move the metric by a similar amount for opposite reasons."""
    shift = investigate_regression("RSI_gender", *CLASSIFIER_SHIFT)
    regression = investigate_regression("RSI_gender", *MODEL_REGRESSION)

    # Both are real movements, comparable in size and direction.
    assert shift.partitioned_delta.exceeds_noise_floor
    assert regression.partitioned_delta.exceeds_noise_floor
    assert shift.partitioned_delta.delta > 0
    assert regression.partitioned_delta.delta > 0

    # The classifier case: the instrument moved, the outputs did not.
    assert shift.classifier_confidence_shift.version_changed
    assert shift.classifier_confidence_shift.shifted
    assert not shift.distribution_shift.shifted
    assert shift.verdict is Verdict.CLASSIFIER_SHIFT_SUSPECTED

    # The model case: the outputs moved, the instrument did not.
    assert not regression.classifier_confidence_shift.version_changed
    assert not regression.classifier_confidence_shift.shifted
    assert regression.distribution_shift.shifted
    assert regression.verdict is Verdict.MODEL_REGRESSION_SUSPECTED


def test_classifier_shift_raises_the_exclusion_rate() -> None:
    """The mechanism behind the false regression is visible in the traces."""
    report = investigate_regression("RSI_gender", *CLASSIFIER_SHIFT)
    delta = report.partitioned_delta
    assert delta.current_exclusion_rate > delta.prior_exclusion_rate + 0.05
    assert report.classifier_confidence_shift.borderline_rate_delta > 0.05
    # And it is concentrated: the prompts at the top of the ranking are the ones
    # whose classifier confidence fell.
    assert report.top_contributing_prompts[0].confidence_delta < 0


def test_confidence_guard_ignores_an_immaterial_drift() -> None:
    """A statistically detectable but operationally tiny drift is not a shift."""
    same = [
        ClassifierCall(classifier="c", classifier_version="1.0", label="man", confidence=0.9)
        for _ in range(200)
    ]
    nudged = [
        ClassifierCall(classifier="c", classifier_version="1.0", label="man", confidence=0.899)
        for _ in range(200)
    ]
    result = compare_confidence_distributions(nudged, same)
    assert not result.version_changed
    assert not result.shifted


def test_a_metric_that_did_not_move_reads_as_no_movement() -> None:
    """HSI_gender is measured by a classifier that never changes version."""
    for current, prior in (CLASSIFIER_SHIFT, MODEL_REGRESSION):
        report = investigate_regression("HSI_gender", current, prior)
        assert report.verdict is Verdict.NO_MOVEMENT
        assert not report.partitioned_delta.exceeds_noise_floor


# ---------------------------------------------------------------------------
# load_traces
# ---------------------------------------------------------------------------


def test_load_traces_honours_attribute_partition() -> None:
    whole = load_traces("soccer_2026_w14", "RSI_gender")
    goalkeepers = load_traces("soccer_2026_w14", "RSI_gender", "role=goalkeeper")

    assert 0 < len(goalkeepers) < len(whole)
    assert {r.attributes["role"] for r in goalkeepers} == {"goalkeeper"}
    assert {r.prompt_cluster for r in goalkeepers} == {"goalkeeper"}


def test_load_traces_honours_an_intersectional_partition() -> None:
    slice_ = load_traces("soccer_2026_w14", "RSI_gender", "role=goalkeeper,skin_tone_band=IV-VI")
    assert len(slice_) > 0
    for record in slice_:
        assert record.attributes["role"] == "goalkeeper"
        assert record.attributes["skin_tone_band"] == "IV-VI"
    # And it is a strict subset of the single-attribute slice.
    assert len(slice_) < len(load_traces("soccer_2026_w14", "RSI_gender", "role=goalkeeper"))


def test_load_traces_filters_by_metric() -> None:
    rsi = load_traces("soccer_2026_w14", "RSI_gender")
    hsi = load_traces("soccer_2026_w14", "HSI_gender")
    assert {r.metric for r in rsi} == {"RSI_gender"}
    assert {r.metric for r in hsi} == {"HSI_gender"}
    assert len(rsi) != len(hsi)


def test_unknown_partition_key_yields_an_empty_set_not_a_silent_full_one() -> None:
    empty = load_traces("soccer_2026_w14", "RSI_gender", "position=libero")
    assert len(empty) == 0


def test_parse_partition_accepts_a_bare_key() -> None:
    assert parse_partition("role=goalkeeper,locale=in") == {"role": "goalkeeper", "locale": "in"}
    assert parse_partition("locale") == {"locale": ""}


def test_traces_expose_outputs_and_classifications() -> None:
    traces = load_traces("soccer_2026_w13", "RSI_gender")
    assert len(traces.outputs) == len(traces.records)
    assert traces.classifications
    assert all(0.0 <= c.confidence <= 1.0 for c in traces.classifications)
    assert traces.classifier_versions == ["person_attributes@3.1"]


def test_investigation_partition_narrows_the_aggregate() -> None:
    whole = investigate_regression("RSI_gender", *CLASSIFIER_SHIFT)
    goalkeepers = investigate_regression("RSI_gender", *CLASSIFIER_SHIFT, "role=goalkeeper")
    assert goalkeepers.attribute_partition == "role=goalkeeper"
    assert len(goalkeepers.partitioned_delta.by_cluster) == 1
    # The slice the movement is concentrated in moves more than the scenario.
    assert goalkeepers.partitioned_delta.delta > whole.partitioned_delta.delta


# ---------------------------------------------------------------------------
# Measurement quality
# ---------------------------------------------------------------------------


def test_targets_parse_from_the_chapters_block() -> None:
    targets = load_targets()
    assert set(targets) == {
        "replicate_stability",
        "sample_size_power",
        "classifier_inter_rater",
    }
    stability = targets["replicate_stability"]
    assert (stability.comparator, stability.threshold) == ("<", 0.02)
    assert stability.scope == "for RSI on soccer_pilot_v1"
    assert stability.frequency == "monthly"
    assert stability.action_on_violation == "Flag for noise-floor recalibration"

    power = targets["sample_size_power"]
    assert (power.comparator, power.threshold) == (">", 0.80)
    assert power.frequency == "per_scenario_design_review"
    assert power.action_on_violation == "Expand prompt set or replicate count"

    kappa = targets["classifier_inter_rater"]
    assert (kappa.comparator, kappa.threshold) == (">", 0.80)
    assert kappa.frequency == "per_classifier_version_change"
    assert kappa.action_on_violation == (
        "Pause automated comparison across the version boundary; human review required"
    )


@pytest.mark.parametrize(
    "metric, breaching, action",
    [
        ("replicate_stability", 0.031, "Flag for noise-floor recalibration"),
        ("sample_size_power", 0.62, "Expand prompt set or replicate count"),
        (
            "classifier_inter_rater",
            0.55,
            "Pause automated comparison across the version boundary; human review required",
        ),
    ],
)
def test_each_target_is_flagged_with_its_action_when_breached(metric, breaching, action) -> None:
    passing = {
        "replicate_stability": 0.012,
        "sample_size_power": 0.91,
        "classifier_inter_rater": 0.88,
    }
    observed = dict(passing)
    observed[metric] = breaching

    assessment = evaluate(observed)
    assert [v.metric for v in assessment.violations] == [metric]
    assert assessment.violations[0].action == action
    assert assessment.violations[0].observed == breaching
    assert not assessment.passed


def test_all_targets_met_passes() -> None:
    assessment = evaluate(
        {"replicate_stability": 0.012, "sample_size_power": 0.91, "classifier_inter_rater": 0.88}
    )
    assert assessment.passed
    assert not assessment.violations
    assert not assessment.not_measured
    assert all(f.passed and f.action == "" for f in assessment.findings)


def test_a_boundary_value_does_not_satisfy_a_strict_target() -> None:
    assessment = evaluate(
        {"replicate_stability": 0.02, "sample_size_power": 0.80, "classifier_inter_rater": 0.80}
    )
    assert {v.metric for v in assessment.violations} == {
        "replicate_stability",
        "sample_size_power",
        "classifier_inter_rater",
    }


def test_a_missing_measurement_is_not_a_pass() -> None:
    assessment = evaluate({"replicate_stability": 0.012})
    assert assessment.not_measured == ["classifier_inter_rater", "sample_size_power"]
    assert not assessment.passed


def test_report_fixtures_evaluate_as_the_narrative_says() -> None:
    breaching = evaluate_report(FIXTURES / "measurement_quality_w14.yaml")
    assert not breaching.passed
    assert {v.metric for v in breaching.violations} == {
        "replicate_stability",
        "classifier_inter_rater",
    }
    assert all(v.action for v in breaching.violations)

    remediated = evaluate_report(FIXTURES / "measurement_quality_w26_post_remediation.yaml")
    assert remediated.passed


def test_inter_rater_kappa_across_the_classifier_boundary_breaches_the_target() -> None:
    """The chapter's point, measured: the two versions do not agree well enough."""
    kappa, n_shared = classifier_inter_rater(
        "soccer_2026_w13", "soccer_2026_w14", "RSI_gender"
    )
    assert n_shared > 0
    assert 0.0 < kappa < 0.80
    assert not load_targets()["classifier_inter_rater"].satisfied_by(kappa)


def test_inter_rater_needs_a_shared_evaluation_set() -> None:
    """Runs that generated their own outputs share no items, so kappa is undefined."""
    kappa, n_shared = classifier_inter_rater(
        "soccer_2026_w20", "soccer_2026_w21", "RSI_gender"
    )
    assert n_shared == 0
    assert kappa != kappa  # NaN


def test_cohen_kappa_corrects_for_chance() -> None:
    perfect = ["a", "b", "a", "b", "c", "c"]
    assert cohen_kappa(perfect, perfect) == pytest.approx(1.0)
    # Two raters that both call everything "a" agree completely and tell us
    # nothing, so kappa is undefined rather than perfect.
    assert cohen_kappa(["a"] * 10, ["a"] * 10) != cohen_kappa(["a"] * 10, ["a"] * 10)
    with pytest.raises(ValueError):
        cohen_kappa(["a", "b"], ["a"])


def test_replicate_stability_needs_more_than_one_run() -> None:
    assert replicate_stability([0.31]) != replicate_stability([0.31])  # NaN
    assert replicate_stability([0.30, 0.32, 0.31]) == pytest.approx(0.01, abs=1e-9)


def test_power_rises_with_sample_size() -> None:
    small = paired_power(0.08, 10)
    large = paired_power(0.08, 80)
    assert 0.0 < small < large <= 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_store(tmp_path, n_prompts: int) -> TraceStore:
    """A two-run store with an arbitrary number of prompts, for the cap test."""
    store = TraceStore(tmp_path / "traces")
    for run_id, offset in (("run_a", 0.0), ("run_b", 0.01)):
        records = []
        for index in range(n_prompts):
            value = 0.20 + 0.005 * index + offset * (index % 5)
            for replicate in range(2):
                records.append(
                    TraceRecord(
                        trace_id=f"{run_id}:p{index}:{replicate}",
                        run_id=run_id,
                        scenario="synthetic",
                        metric="RSI_gender",
                        prompt_id=f"p{index:02d}",
                        prompt=f"prompt {index}",
                        prompt_cluster="synthetic",
                        attributes={"role": "synthetic"},
                        replicate=replicate,
                        model_call=ModelCall(
                            model_version="m1",
                            output_id=f"{run_id}:p{index}:{replicate}",
                            reference_probe="probe",
                            reference_probe_label="man",
                        ),
                        classifier_calls=[
                            ClassifierCall(
                                classifier="c",
                                classifier_version="1.0",
                                label="man",
                                confidence=0.9,
                            )
                        ],
                        metric_contribution=value,
                    )
                )
        store.write(run_id, records)
    return store
