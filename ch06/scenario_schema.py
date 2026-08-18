"""A validatable schema for Chapter 6 scenario templates.

Chapter 6 builds prompt/scenario templates in YAML. Both reviewers asked for a
formal schema readers can validate their own scenario files against, and a way
to tie those templates back to the Chapter 5 code. This module is that schema:
a small pydantic model plus a loader and a JSON Schema export.

    from ch06.scenario_schema import load_scenario, json_schema
    scenario = load_scenario("ch06/examples/education_dialect.yaml")

It is deliberately self-contained (pydantic + PyYAML) so it validates a scenario
file without needing the full library installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

FairnessDimension = Literal["representational", "distributional", "counterfactual", "procedural"]


class Scenario(BaseModel):
    """A fairness scenario template.

    Unknown top-level keys are rejected (``extra="forbid"``) so that typos such
    as ``harm_typ`` or ``prompt_templat`` fail loudly at load time rather than
    surfacing as a confusing error deep inside prompt generation.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # Required
    id: str
    domain: str
    prompt_template: str

    # Classification
    fairness_dimension: FairnessDimension | None = None
    harm_type: list[str] = Field(default_factory=list)

    # Generation / evaluation
    attribute_axes: dict[str, list[Any]] | None = None
    sensitive_attributes: list[str] | None = None
    evaluation_targets: list[str] | None = None
    evaluation_method: str | None = None
    reference_distributions: dict[str, Any] | None = None
    replicates_per_prompt: int | None = None
    fairness_expectation: str | None = None

    # Coverage / localization metadata (Chapter 6). `register` is exposed under
    # the field name `language_register` to avoid shadowing a BaseModel method,
    # while YAML files keep using the natural `register:` key via the alias.
    locale: str | None = None
    language_register: str | None = Field(default=None, alias="register")
    localization_type: str | None = None

    # Housekeeping
    version: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    created_by: str | None = None
    last_reviewed: str | None = None

    @field_validator("harm_type", mode="before")
    @classmethod
    def _normalize_harm_type(cls, v: Any) -> list[str]:
        """Accept a list, or a ``"a + b"`` / ``"a, b"`` string, normalize to a list.

        The Chapter 6 drafts wrote harm_type as a concatenated string with a
        plus sign; a real schema stores a sequence. This validator accepts both
        so old templates still load, but always yields a clean list.
        """
        if v is None:
            return []
        if isinstance(v, str):
            return [h.strip() for h in v.replace("+", ",").split(",") if h.strip()]
        if isinstance(v, list):
            return [str(h).strip() for h in v]
        raise ValueError("harm_type must be a string or a list of strings")


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate a scenario YAML file.

    Templates may nest their fields under a top-level ``scenario:`` key (as the
    book listings do); this loader unwraps that automatically.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and isinstance(data.get("scenario"), dict):
        data = data["scenario"]
    return Scenario.model_validate(data)


def json_schema() -> dict[str, Any]:
    """Return the JSON Schema for a scenario, for editor validation or CI."""
    return Scenario.model_json_schema()
