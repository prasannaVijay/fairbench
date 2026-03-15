"""Base scenario loading functionality."""

from pathlib import Path
from typing import Any

import yaml

from fairbench.core.exceptions import ScenarioError
from fairbench.core.types import (
    CounterfactualGroup,
    CounterfactualVariant,
    FairnessDimension,
    Scenario,
    ScenarioSet,
    SensitiveAttribute,
)


class ScenarioLoader:
    """Loads scenarios from YAML/JSON files."""

    @staticmethod
    def load_file(path: Path | str) -> ScenarioSet:
        """Load a scenario set from a file.

        Args:
            path: Path to the scenario file (YAML or JSON).

        Returns:
            The loaded ScenarioSet.

        Raises:
            ScenarioError: If the file cannot be loaded or is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise ScenarioError(f"Scenario file not found: {path}")

        try:
            with open(path) as f:
                if path.suffix in (".yaml", ".yml"):
                    data = yaml.safe_load(f)
                elif path.suffix == ".json":
                    import json

                    data = json.load(f)
                else:
                    raise ScenarioError(f"Unsupported file format: {path.suffix}")
        except (yaml.YAMLError, ValueError) as e:
            raise ScenarioError(f"Failed to parse scenario file: {e}") from e

        return ScenarioLoader.parse(data)

    @staticmethod
    def parse(data: dict[str, Any]) -> ScenarioSet:
        """Parse a scenario set from a dictionary.

        Args:
            data: The raw scenario data.

        Returns:
            The parsed ScenarioSet.

        Raises:
            ScenarioError: If the data is invalid.
        """
        try:
            scenarios = []
            for scenario_data in data.get("scenarios", []):
                scenario = ScenarioLoader._parse_scenario(scenario_data)
                scenarios.append(scenario)

            dimensions = [
                FairnessDimension(d) for d in data.get("dimensions", [])
            ]

            return ScenarioSet(
                name=data.get("name", "unnamed"),
                version=data.get("version", "1.0"),
                description=data.get("description"),
                dimensions=dimensions,
                scenarios=scenarios,
                metadata=data.get("metadata", {}),
            )
        except Exception as e:
            raise ScenarioError(f"Failed to parse scenario set: {e}") from e

    @staticmethod
    def _parse_scenario(data: dict[str, Any]) -> Scenario:
        """Parse a single scenario from a dictionary."""
        counterfactuals = []
        for cf_data in data.get("counterfactuals", []):
            cf_group = ScenarioLoader._parse_counterfactual_group(cf_data)
            counterfactuals.append(cf_group)

        dimensions = [
            FairnessDimension(d) for d in data.get("dimensions", [])
        ]

        return Scenario(
            id=data["id"],
            prompt=data["prompt"],
            description=data.get("description"),
            dimensions=dimensions,
            counterfactuals=counterfactuals,
            expected_behavior=data.get("expected_behavior"),
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def _parse_counterfactual_group(data: dict[str, Any]) -> CounterfactualGroup:
        """Parse a counterfactual group from a dictionary."""
        # Try to parse as SensitiveAttribute enum, fall back to string
        attribute_str = data["attribute"]
        try:
            attribute: SensitiveAttribute | str = SensitiveAttribute(attribute_str)
        except ValueError:
            attribute = attribute_str

        variants = []
        for variant_data in data.get("variants", []):
            variant = CounterfactualVariant(
                prompt=variant_data["prompt"],
                attribute_value=variant_data.get("value", ""),
                description=variant_data.get("description"),
            )
            variants.append(variant)

        return CounterfactualGroup(attribute=attribute, variants=variants)

    @staticmethod
    def load_directory(path: Path | str) -> list[ScenarioSet]:
        """Load all scenario sets from a directory.

        Args:
            path: Path to the directory containing scenario files.

        Returns:
            List of loaded ScenarioSets.
        """
        path = Path(path)
        if not path.is_dir():
            raise ScenarioError(f"Not a directory: {path}")

        scenario_sets = []
        for file_path in path.glob("*.yaml"):
            scenario_sets.append(ScenarioLoader.load_file(file_path))
        for file_path in path.glob("*.yml"):
            scenario_sets.append(ScenarioLoader.load_file(file_path))
        for file_path in path.glob("*.json"):
            scenario_sets.append(ScenarioLoader.load_file(file_path))

        return scenario_sets
