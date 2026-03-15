"""Tests for scenario loading and registry."""

import pytest
from pathlib import Path

from fairbench.core.types import FairnessDimension, Scenario, ScenarioSet
from fairbench.scenarios.base import ScenarioLoader
from fairbench.scenarios.registry import ScenarioRegistry


class TestScenarioLoader:
    """Tests for ScenarioLoader."""

    def test_parse_simple_scenario(self) -> None:
        """Test parsing a simple scenario set."""
        data = {
            "name": "test_set",
            "version": "1.0",
            "description": "Test scenarios",
            "dimensions": ["representational"],
            "scenarios": [
                {
                    "id": "test_1",
                    "prompt": "Test prompt",
                    "description": "A test scenario",
                }
            ],
        }

        result = ScenarioLoader.parse(data)

        assert result.name == "test_set"
        assert result.version == "1.0"
        assert len(result.scenarios) == 1
        assert result.scenarios[0].id == "test_1"
        assert result.scenarios[0].prompt == "Test prompt"
        assert FairnessDimension.REPRESENTATIONAL in result.dimensions

    def test_parse_scenario_with_counterfactuals(self) -> None:
        """Test parsing scenarios with counterfactual variants."""
        data = {
            "name": "cf_test",
            "scenarios": [
                {
                    "id": "cf_scenario",
                    "prompt": "Base prompt",
                    "counterfactuals": [
                        {
                            "attribute": "gender",
                            "variants": [
                                {"prompt": "Male variant", "value": "male"},
                                {"prompt": "Female variant", "value": "female"},
                            ],
                        }
                    ],
                }
            ],
        }

        result = ScenarioLoader.parse(data)
        scenario = result.scenarios[0]

        assert len(scenario.counterfactuals) == 1
        assert scenario.counterfactuals[0].attribute == "gender"
        assert len(scenario.counterfactuals[0].variants) == 2


class TestScenarioRegistry:
    """Tests for ScenarioRegistry."""

    def test_register_and_retrieve(self) -> None:
        """Test registering and retrieving scenario sets."""
        registry = ScenarioRegistry()

        scenario_set = ScenarioSet(
            name="test_registry",
            scenarios=[
                Scenario(id="s1", prompt="Prompt 1"),
                Scenario(id="s2", prompt="Prompt 2"),
            ],
        )

        registry.register(scenario_set)

        assert "test_registry" in registry.list_sets()
        retrieved = registry.get_set("test_registry")
        assert len(retrieved.scenarios) == 2

    def test_get_scenario_by_id(self) -> None:
        """Test retrieving individual scenarios by ID."""
        registry = ScenarioRegistry()

        scenario_set = ScenarioSet(
            name="test_set",
            scenarios=[
                Scenario(id="unique_id", prompt="Test prompt"),
            ],
        )

        registry.register(scenario_set)

        # Should work with full ID
        scenario = registry.get_scenario("test_set:unique_id")
        assert scenario.id == "unique_id"

        # Should also work with short ID if unique
        scenario = registry.get_scenario("unique_id")
        assert scenario.id == "unique_id"

    def test_load_builtin_scenarios(self) -> None:
        """Test that built-in scenarios can be loaded."""
        registry = ScenarioRegistry()
        builtin = registry.load_builtin()

        # Should have loaded the built-in scenario files
        assert len(builtin) > 0
        assert any("gender" in s.name for s in builtin)
