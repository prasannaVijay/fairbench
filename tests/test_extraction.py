"""Metrics score what the model produced, not what the prompts requested."""

from uuid import uuid4

import pytest

from fairbench_genai.core.exceptions import MetricError
from fairbench_genai.core.types import (
    Distribution,
    EvaluatedOutput,
    GeneratedOutput,
    GenerationConfig,
    ModelInfo,
)
from fairbench_genai.metrics.extraction import (
    DETECTED,
    REQUESTED,
    available_attributes,
    extract_categories,
)
from fairbench_genai.metrics.sar import StereotypeAmplificationRatio


def make_output(
    requested: str | None = None,
    attribute: str | None = None,
    detected: dict[str, list[str]] | None = None,
) -> EvaluatedOutput:
    return EvaluatedOutput(
        id=uuid4(),
        output=GeneratedOutput(
            text="text",
            prompt="prompt",
            model_info=ModelInfo(name="test", provider="test"),
            generation_config=GenerationConfig(),
        ),
        scenario_id="s1",
        is_counterfactual=requested is not None,
        counterfactual_attribute=attribute,
        counterfactual_value=requested,
        detected_entities=detected or {},
    )


class TestExtraction:
    def test_infers_the_only_axis_present(self) -> None:
        outputs = [make_output(detected={"gender_signal": ["masculine"]})]

        extracted = extract_categories(outputs)

        assert extracted.attribute == "gender"
        assert extracted.labels == ["male"]

    def test_refuses_to_mix_two_axes(self) -> None:
        outputs = [
            make_output(
                detected={
                    "gender_signal": ["masculine"],
                    "name_origins": ["east_asian"],
                }
            )
        ]

        with pytest.raises(MetricError, match="more than one attribute axis"):
            extract_categories(outputs)

        assert extract_categories(outputs, attribute="gender").labels == ["male"]
        assert extract_categories(outputs, attribute="ethnicity").labels == [
            "east_asian"
        ]

    def test_reports_when_nothing_was_detected(self) -> None:
        with pytest.raises(MetricError, match="No detected demographic categories"):
            extract_categories([make_output(requested="male", attribute="gender")])

    def test_requested_source_reads_the_generated_variant(self) -> None:
        outputs = [make_output(requested="chinese", attribute="ethnicity")]

        extracted = extract_categories(outputs, source=REQUESTED)

        assert extracted.attribute == "ethnicity"
        assert extracted.labels == ["east_asian"]

    def test_available_attributes_sees_both_sources(self) -> None:
        outputs = [
            make_output(
                requested="male",
                attribute="gender",
                detected={"name_origins": ["white_western"]},
            )
        ]

        assert available_attributes(outputs, DETECTED) == {"ethnicity"}
        assert available_attributes(outputs, REQUESTED) == {"gender"}

    def test_absent_categories_are_counted_as_zero(self) -> None:
        outputs = [make_output(detected={"name_origins": ["east_asian"]})]

        counts = extract_categories(outputs).counts()

        assert counts["east_asian"] == 1
        assert counts["white_western"] == 0
        assert len(counts) == 6


class TestSARReadsTheModel:
    """The behaviour change that makes SAR describe the model at all."""

    def test_balanced_prompts_with_a_skewed_model(self) -> None:
        # The prompt set is balanced by construction: two male variants and
        # two female. The model answers with masculine pronouns three times
        # out of four.
        detected = ["masculine", "masculine", "masculine", "feminine"]
        requested = ["male", "female", "male", "female"]
        outputs = [
            make_output(
                requested=r,
                attribute="gender",
                detected={"gender_signal": [d]},
            )
            for r, d in zip(requested, detected)
        ]
        baseline = Distribution({"male": 0.5, "female": 0.5})

        on_the_model = StereotypeAmplificationRatio().compute(outputs, baseline)
        on_the_prompts = StereotypeAmplificationRatio(
            category_source=REQUESTED
        ).compute(outputs, baseline)

        assert on_the_model.value == pytest.approx(3.0)
        assert on_the_prompts.value == pytest.approx(1.0)
        assert on_the_model.details["category_source"] == "detected"
