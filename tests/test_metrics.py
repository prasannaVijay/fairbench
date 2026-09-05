"""Tests for fairness metrics."""

import math
from uuid import uuid4

import pytest

from fairbench_genai.core.types import (
    Distribution,
    EvaluatedOutput,
    GeneratedOutput,
    GenerationConfig,
    ModelInfo,
    SentimentScores,
    ToxicityScores,
)
from fairbench_genai.metrics.cds import CounterfactualDivergenceScore
from fairbench_genai.metrics.hsi import HarmSeverityIndex
from fairbench_genai.metrics.ode import OutputDiversityEntropy
from fairbench_genai.metrics.rsi import RepresentationSkewIndex
from fairbench_genai.metrics.sar import StereotypeAmplificationRatio


def make_output(
    text: str,
    scenario_id: str,
    is_cf: bool = False,
    cf_attr: str | None = None,
    cf_value: str | None = None,
    embedding: list[float] | None = None,
    toxicity: ToxicityScores | None = None,
) -> EvaluatedOutput:
    """Helper to create test outputs."""
    return EvaluatedOutput(
        id=uuid4(),
        output=GeneratedOutput(
            text=text,
            prompt="test prompt",
            model_info=ModelInfo(name="test", provider="test"),
            generation_config=GenerationConfig(),
        ),
        scenario_id=scenario_id,
        is_counterfactual=is_cf,
        counterfactual_attribute=cf_attr,
        counterfactual_value=cf_value,
        embedding=embedding,
        toxicity=toxicity,
    )


class TestCDS:
    """Tests for Counterfactual Divergence Score."""

    def test_identical_outputs_low_cds(self) -> None:
        """Identical outputs should have low CDS."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        outputs = [
            make_output("Output 1", "s1", embedding=embedding),
            make_output("Output 2", "s1", is_cf=True, cf_attr="gender", cf_value="female", embedding=embedding),
        ]

        cds = CounterfactualDivergenceScore()
        result = cds.compute(outputs)

        assert result.value < 0.1  # Should be very low

    def test_different_outputs_higher_cds(self) -> None:
        """Different outputs should have higher CDS."""
        outputs = [
            make_output("Output 1", "s1", embedding=[1.0, 0.0, 0.0, 0.0, 0.0]),
            make_output("Output 2", "s1", is_cf=True, cf_attr="gender", cf_value="female",
                       embedding=[0.0, 1.0, 0.0, 0.0, 0.0]),
        ]

        cds = CounterfactualDivergenceScore()
        result = cds.compute(outputs)

        assert result.value > 0.5  # Should be higher due to different embeddings


class TestHSI:
    """Tests for Harm Severity Index."""

    def test_no_toxicity_zero_hsi(self) -> None:
        """Outputs with no toxicity should have HSI of 0."""
        outputs = [
            make_output("Clean text", "s1", toxicity=ToxicityScores()),
            make_output("Also clean", "s2", toxicity=ToxicityScores()),
        ]

        hsi = HarmSeverityIndex()
        result = hsi.compute(outputs)

        assert result.value == 0.0

    def test_toxic_outputs_positive_hsi(self) -> None:
        """Outputs with toxicity should have positive HSI."""
        outputs = [
            make_output(
                "Toxic text",
                "s1",
                toxicity=ToxicityScores(toxicity=0.8, identity_attack=0.6),
            ),
            make_output("Clean text", "s2", toxicity=ToxicityScores()),
        ]

        hsi = HarmSeverityIndex()
        result = hsi.compute(outputs)

        assert result.value > 0.0
        assert result.details["total_harmful"] == 1


class TestRSI:
    """Tests for Representation Skew Index."""

    def test_uniform_distribution_low_rsi(self) -> None:
        """Uniform output distribution should have low RSI against uniform baseline."""
        outputs = [
            make_output("A", "s1", is_cf=True, cf_attr="gender", cf_value="male"),
            make_output("B", "s1", is_cf=True, cf_attr="gender", cf_value="female"),
        ]

        rsi = RepresentationSkewIndex()
        baseline = Distribution({"male": 0.5, "female": 0.5})
        result = rsi.compute(outputs, baseline)

        assert result.value < 0.1

    def test_skewed_distribution_higher_rsi(self) -> None:
        """Skewed distribution should have higher RSI."""
        # 3 male, 1 female = 75/25 split
        outputs = [
            make_output("A", "s1", is_cf=True, cf_attr="gender", cf_value="male"),
            make_output("B", "s1", is_cf=True, cf_attr="gender", cf_value="male"),
            make_output("C", "s1", is_cf=True, cf_attr="gender", cf_value="male"),
            make_output("D", "s1", is_cf=True, cf_attr="gender", cf_value="female"),
        ]

        rsi = RepresentationSkewIndex()
        baseline = Distribution({"male": 0.5, "female": 0.5})
        result = rsi.compute(outputs, baseline)

        # A 75/25 split is a mild skew: the JSD-based RSI is small but strictly
        # positive (~0.049 in log base 2), and well above the 0.0 a balanced
        # split produces.
        assert 0.0 < result.value < 0.1  # Should show skew

    def test_disjoint_distributions_reach_the_bound(self) -> None:
        """RSI is reported in log base 2, so disjoint distributions score 1.0."""
        outputs = [
            make_output("A", "s1", is_cf=True, cf_attr="gender", cf_value="male"),
            make_output("B", "s1", is_cf=True, cf_attr="gender", cf_value="male"),
        ]

        rsi = RepresentationSkewIndex()
        baseline = Distribution({"male": 0.0, "female": 1.0})
        result = rsi.compute(outputs, baseline)

        assert result.value == pytest.approx(1.0, abs=1e-6)

    def test_result_records_the_log_base(self) -> None:
        """Scorecards must be able to tell which scale produced a value."""
        outputs = [
            make_output("A", "s1", is_cf=True, cf_attr="gender", cf_value="male"),
            make_output("B", "s1", is_cf=True, cf_attr="gender", cf_value="female"),
        ]

        rsi = RepresentationSkewIndex()
        result = rsi.compute(outputs, Distribution({"male": 0.5, "female": 0.5}))

        assert result.details["log_base"] == 2

    def test_bands_are_derived_from_the_natural_log_thresholds(self) -> None:
        """Rescaling the bands must leave every verdict unchanged."""
        rsi = RepresentationSkewIndex()
        ln2 = math.log(2)

        for legacy_value, expected in (
            (0.10, "Pass"),
            (0.15, "Pass"),
            (0.20, "Watch"),
            (0.25, "Watch"),
            (0.30, "Flag"),
            (0.40, "Flag"),
            (0.55, "Fail"),
        ):
            band = rsi.interpret_value(legacy_value / ln2).split(" - ")[0]
            assert band == expected, f"{legacy_value} became {band}"


class TestODE:
    """Tests for Output Diversity Entropy."""

    def test_diverse_outputs_high_ode(self) -> None:
        """Diverse outputs should have high ODE."""
        outputs = [
            make_output("Unique text 1", "s1", is_cf=True, cf_value="a"),
            make_output("Unique text 2", "s2", is_cf=True, cf_value="b"),
            make_output("Unique text 3", "s3", is_cf=True, cf_value="c"),
            make_output("Unique text 4", "s4", is_cf=True, cf_value="d"),
        ]

        ode = OutputDiversityEntropy(diversity_method="attribute_counts")
        result = ode.compute(outputs)

        assert result.value > 0.8  # High diversity

    def test_repeated_outputs_low_ode(self) -> None:
        """Repeated outputs should have low ODE."""
        # 5 of same value, 1 different = very skewed
        outputs = [
            make_output("Same text", "s1", is_cf=True, cf_value="a"),
            make_output("Same text", "s2", is_cf=True, cf_value="a"),
            make_output("Same text", "s3", is_cf=True, cf_value="a"),
            make_output("Same text", "s4", is_cf=True, cf_value="a"),
            make_output("Same text", "s5", is_cf=True, cf_value="a"),
            make_output("Different", "s6", is_cf=True, cf_value="b"),
        ]

        ode = OutputDiversityEntropy(diversity_method="attribute_counts")
        result = ode.compute(outputs)

        assert result.value < 0.7  # Lower diversity due to concentration
