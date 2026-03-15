"""Scenario registry for FAIRBench."""

from pathlib import Path

from fairbench.core.exceptions import ScenarioError
from fairbench.core.types import FairnessDimension, Scenario, ScenarioSet, SensitiveAttribute
from fairbench.scenarios.base import ScenarioLoader


class ScenarioRegistry:
    """Registry for scenario sets.

    The registry provides:
    - Loading scenarios from files and directories
    - Querying scenarios by dimension, attribute, or ID
    - Access to built-in scenario sets
    """

    def __init__(self) -> None:
        self._sets: dict[str, ScenarioSet] = {}
        self._scenarios: dict[str, Scenario] = {}

    def register(self, scenario_set: ScenarioSet) -> None:
        """Register a scenario set.

        Args:
            scenario_set: The scenario set to register.

        Raises:
            ScenarioError: If a set with the same name already exists.
        """
        if scenario_set.name in self._sets:
            raise ScenarioError(f"Scenario set already registered: {scenario_set.name}")

        self._sets[scenario_set.name] = scenario_set

        # Index individual scenarios
        for scenario in scenario_set.scenarios:
            full_id = f"{scenario_set.name}:{scenario.id}"
            self._scenarios[full_id] = scenario
            # Also allow lookup by short ID if unique
            if scenario.id not in self._scenarios:
                self._scenarios[scenario.id] = scenario

    def load_file(self, path: Path | str) -> ScenarioSet:
        """Load and register a scenario set from a file.

        Args:
            path: Path to the scenario file.

        Returns:
            The loaded and registered ScenarioSet.
        """
        scenario_set = ScenarioLoader.load_file(path)
        self.register(scenario_set)
        return scenario_set

    def load_directory(self, path: Path | str) -> list[ScenarioSet]:
        """Load and register all scenario sets from a directory.

        Args:
            path: Path to the directory.

        Returns:
            List of loaded and registered ScenarioSets.
        """
        scenario_sets = ScenarioLoader.load_directory(path)
        for scenario_set in scenario_sets:
            self.register(scenario_set)
        return scenario_sets

    def load_builtin(self) -> list[ScenarioSet]:
        """Load built-in scenario sets.

        Returns:
            List of built-in ScenarioSets.
        """
        builtin_dir = Path(__file__).parent / "builtin"
        if builtin_dir.exists():
            return self.load_directory(builtin_dir)
        return []

    def get_set(self, name: str) -> ScenarioSet:
        """Get a scenario set by name.

        Args:
            name: Name of the scenario set.

        Returns:
            The scenario set.

        Raises:
            ScenarioError: If the set is not found.
        """
        if name not in self._sets:
            raise ScenarioError(f"Scenario set not found: {name}")
        return self._sets[name]

    def get_scenario(self, scenario_id: str) -> Scenario:
        """Get a scenario by ID.

        Args:
            scenario_id: ID of the scenario (can be "set:id" or just "id").

        Returns:
            The scenario.

        Raises:
            ScenarioError: If the scenario is not found.
        """
        if scenario_id not in self._scenarios:
            raise ScenarioError(f"Scenario not found: {scenario_id}")
        return self._scenarios[scenario_id]

    def list_sets(self) -> list[str]:
        """List all registered scenario set names."""
        return list(self._sets.keys())

    def list_scenarios(self, set_name: str | None = None) -> list[str]:
        """List scenario IDs, optionally filtered by set.

        Args:
            set_name: If provided, only list scenarios from this set.

        Returns:
            List of scenario IDs.
        """
        if set_name is not None:
            scenario_set = self.get_set(set_name)
            return [s.id for s in scenario_set.scenarios]
        return list(self._scenarios.keys())

    def query(
        self,
        dimension: FairnessDimension | None = None,
        attribute: SensitiveAttribute | str | None = None,
        set_name: str | None = None,
    ) -> list[Scenario]:
        """Query scenarios by criteria.

        Args:
            dimension: Filter by fairness dimension.
            attribute: Filter by sensitive attribute tested.
            set_name: Filter by scenario set name.

        Returns:
            List of matching scenarios.
        """
        results = []

        sets_to_search = (
            [self.get_set(set_name)] if set_name else list(self._sets.values())
        )

        for scenario_set in sets_to_search:
            for scenario in scenario_set.scenarios:
                # Check dimension filter
                if dimension is not None:
                    scenario_dims = scenario.dimensions or scenario_set.dimensions
                    if dimension not in scenario_dims:
                        continue

                # Check attribute filter
                if attribute is not None:
                    has_attribute = any(
                        cf.attribute == attribute for cf in scenario.counterfactuals
                    )
                    if not has_attribute:
                        continue

                results.append(scenario)

        return results

    def get_all_scenarios(self) -> list[Scenario]:
        """Get all registered scenarios."""
        scenarios = []
        for scenario_set in self._sets.values():
            scenarios.extend(scenario_set.scenarios)
        return scenarios


# Global registry instance
_registry: ScenarioRegistry | None = None


def get_registry() -> ScenarioRegistry:
    """Get the global scenario registry."""
    global _registry
    if _registry is None:
        _registry = ScenarioRegistry()
        _registry.load_builtin()
    return _registry


def reset_registry() -> None:
    """Reset the global registry."""
    global _registry
    _registry = None
