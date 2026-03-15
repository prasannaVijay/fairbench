"""Baseline distribution registry."""

from fairbench.core.exceptions import BaselineNotFoundError
from fairbench.core.types import Distribution


class BaselineRegistry:
    """Registry of baseline distributions for fairness metrics.

    Baselines represent expected or fair distributions that actual
    model outputs are compared against.

    Types of baselines:
    - Uniform: Equal probability across all categories
    - Real-world: Based on actual demographic data
    - Custom: User-defined distributions
    """

    def __init__(self) -> None:
        self._baselines: dict[str, Distribution] = {}
        self._load_builtin()

    def _load_builtin(self) -> None:
        """Load built-in baseline distributions."""
        # Gender baselines
        self.register(
            "gender_binary_uniform",
            Distribution({"male": 0.5, "female": 0.5}),
        )
        self.register(
            "gender_inclusive_uniform",
            Distribution({"male": 0.33, "female": 0.33, "non-binary": 0.34}),
        )

        # US Census-based baselines (2020)
        self.register(
            "us_gender_census",
            Distribution({"male": 0.494, "female": 0.506}),
        )
        self.register(
            "us_race_census_2020",
            Distribution({
                "white": 0.576,
                "hispanic": 0.187,
                "black": 0.121,
                "asian": 0.059,
                "multiracial": 0.044,
                "other": 0.013,
            }),
        )

        # Occupation-specific baselines (US Bureau of Labor Statistics)
        # These show actual distributions, useful for SAR calculation
        self.register(
            "us_doctors_gender",
            Distribution({"male": 0.637, "female": 0.363}),
        )
        self.register(
            "us_nurses_gender",
            Distribution({"male": 0.121, "female": 0.879}),
        )
        self.register(
            "us_engineers_gender",
            Distribution({"male": 0.84, "female": 0.16}),
        )
        self.register(
            "us_teachers_gender",
            Distribution({"male": 0.24, "female": 0.76}),
        )
        self.register(
            "us_ceos_gender",
            Distribution({"male": 0.896, "female": 0.104}),
        )
        self.register(
            "us_pilots_gender",
            Distribution({"male": 0.947, "female": 0.053}),
        )

        # Age distribution baselines
        self.register(
            "age_groups_uniform",
            Distribution({
                "young": 0.33,
                "middle_aged": 0.34,
                "elderly": 0.33,
            }),
        )

    def register(self, name: str, distribution: Distribution) -> None:
        """Register a baseline distribution.

        Args:
            name: Name for the baseline.
            distribution: The distribution to register.
        """
        self._baselines[name] = distribution

    def get(self, name: str) -> Distribution:
        """Get a baseline distribution by name.

        Args:
            name: Name of the baseline.

        Returns:
            The distribution.

        Raises:
            BaselineNotFoundError: If not found.
        """
        if name not in self._baselines:
            raise BaselineNotFoundError(name)
        return self._baselines[name]

    def get_or_uniform(
        self, name: str | None, categories: list[str]
    ) -> Distribution:
        """Get a baseline or create uniform distribution.

        Args:
            name: Optional baseline name.
            categories: Categories for uniform fallback.

        Returns:
            The requested or uniform distribution.
        """
        if name:
            try:
                return self.get(name)
            except BaselineNotFoundError:
                pass
        return Distribution.uniform(categories)

    def list(self) -> list[str]:
        """List all registered baseline names."""
        return list(self._baselines.keys())

    def has(self, name: str) -> bool:
        """Check if a baseline exists."""
        return name in self._baselines


# Global registry instance
_registry: BaselineRegistry | None = None


def get_baseline_registry() -> BaselineRegistry:
    """Get the global baseline registry."""
    global _registry
    if _registry is None:
        _registry = BaselineRegistry()
    return _registry


def reset_baseline_registry() -> None:
    """Reset the global baseline registry."""
    global _registry
    _registry = None
