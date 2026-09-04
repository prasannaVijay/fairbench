"""Tests for the Chapter 6 scenario schema and example templates."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch06"))
from scenario_schema import Scenario, json_schema, load_scenario  # noqa: E402

CH06 = Path(__file__).resolve().parent.parent / "ch06"
# Chapter 6 prints the transcript of the `examples/*.yaml` glob and closes it on
# "3/3 valid", so `examples/` holds exactly the three files the page lists and the
# healthcare template sits in `templates/`. Both directories are validated here.
EXAMPLES = sorted(CH06.joinpath("examples").glob("*.yaml")) + sorted(
    CH06.joinpath("templates").glob("*.yaml")
)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_examples_validate(path: Path) -> None:
    scenario = load_scenario(path)
    assert scenario.id
    assert scenario.domain
    assert scenario.prompt_template
    assert isinstance(scenario.harm_type, list)


def test_harm_type_string_is_normalized_to_list() -> None:
    s = Scenario(id="x", domain="d", prompt_template="p", harm_type="erasure + misrecognition")
    assert s.harm_type == ["erasure", "misrecognition"]


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(Exception):
        Scenario(id="x", domain="d", prompt_template="p", harm_typ="oops")


def test_bad_fairness_dimension_is_rejected() -> None:
    with pytest.raises(Exception):
        Scenario(id="x", domain="d", prompt_template="p", fairness_dimension="not_a_dimension")


def test_json_schema_exports() -> None:
    schema = json_schema()
    assert schema["title"] == "Scenario"
    assert "prompt_template" in schema["properties"]
