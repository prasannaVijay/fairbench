"""Scenario configuration loading and validation for the Chapter 8 pilot.

The scenario file is the contract between the benchmark and the model under
test. ``ScenarioStore`` reads it, checks it, and expands it into the concrete
list of prompt variants the run will send. Validation happens before a single
image is generated, because a misconfigured scenario that fails after 840 API
calls is expensive and demoralising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml


class ScenarioValidationError(ValueError):
    """Raised when a scenario file does not satisfy the schema."""


@dataclass(frozen=True)
class ReferenceDistribution:
    """A named reference distribution and the reasoning that justifies it.

    A reference distribution is a normative claim about what an equitable spread
    of outputs looks like. It is carried around with its rationale and its
    provenance so that a disputed metric can be argued about at the level where
    the disagreement actually lives.
    """

    name: str
    probabilities: dict[str, float]
    rationale: str = ""
    provenance: str = ""

    def share(self, category: str) -> float:
        return self.probabilities.get(category, 0.0)


@dataclass(frozen=True)
class SensitiveAttribute:
    """One attribute the run varies and measures."""

    name: str
    reference_distribution: ReferenceDistribution
    classifier: str
    categories: list[str]
    counterfactual_modifiers: list[str]
    ambiguous_label: str = "ambiguous"


@dataclass(frozen=True)
class PromptVariant:
    """One prompt the benchmark will send, with the metadata that follows it.

    The metadata travels with the generated image all the way to the artifact
    store, which is what lets a metric value be traced back to the exact prompt
    and replicate behind it.
    """

    id: str
    text: str
    role: str
    action: str
    condition: str
    attribute: str | None
    modifier: str | None
    replicates: int

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "variant_id": self.id,
            "role": self.role,
            "action": self.action,
            "condition": self.condition,
            "attribute": self.attribute,
            "modifier": self.modifier,
            "prompt": self.text,
        }


@dataclass(frozen=True)
class BasePrompt:
    id: str
    role: str
    text: str
    replicates: int
    actions: list[str]


@dataclass
class Scenario:
    """A validated scenario, expanded into prompt variants."""

    id: str
    version: str
    domain: str
    sensitive_attributes: list[SensitiveAttribute]
    base_prompts: list[BasePrompt]
    neutral_prompt_template: str
    human_review_fraction: float = 0.15
    source_path: Path | None = None
    _variants: list[PromptVariant] = field(default_factory=list, repr=False)

    def prompt_variants(self) -> list[PromptVariant]:
        return list(self._variants)

    def total_prompt_count(self) -> int:
        """Number of distinct prompt variants, before replicates."""
        return len(self._variants)

    def total_image_count(self) -> int:
        """Number of images the run will generate, replicates included."""
        return sum(v.replicates for v in self._variants)

    def attribute(self, name: str) -> SensitiveAttribute:
        for a in self.sensitive_attributes:
            if a.name == name:
                return a
        raise KeyError(name)

    def __iter__(self) -> Iterator[PromptVariant]:
        return iter(self._variants)


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ScenarioValidationError(f"{where}: missing required field {key!r}")
    return mapping[key]


class ScenarioStore:
    """Reads, validates and expands a scenario configuration file."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)

    def load(self, validate: bool = True) -> Scenario:
        """Load the scenario, optionally checking it against the schema first.

        ``validate=False`` exists for the case where a caller has already
        validated the file and wants to skip the second pass; it is not a way to
        run an unchecked configuration against a paid endpoint.
        """
        if not self.config_path.exists():
            raise ScenarioValidationError(f"scenario file not found: {self.config_path}")
        raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ScenarioValidationError("scenario file must contain a mapping at the top level")
        if validate:
            self._validate(raw)
        return self._build(raw)

    # -- validation -------------------------------------------------------

    def _validate(self, raw: dict[str, Any]) -> None:
        where = str(self.config_path)
        for key in ("scenario_id", "version", "domain", "sensitive_attributes", "base_prompts"):
            _require(raw, key, where)

        variants = raw.get("counterfactual_variants") or {}
        if not isinstance(variants, dict):
            raise ScenarioValidationError(f"{where}: counterfactual_variants must be a mapping")

        seen: set[str] = set()
        for attr in raw["sensitive_attributes"]:
            name = _require(attr, "name", f"{where}: sensitive_attributes")
            if name in seen:
                raise ScenarioValidationError(f"{where}: duplicate sensitive attribute {name!r}")
            seen.add(name)
            _require(attr, "reference_distribution", f"{where}: attribute {name}")
            _require(attr, "classifier", f"{where}: attribute {name}")
            categories = _require(attr, "categories", f"{where}: attribute {name}")
            if len(categories) < 2:
                raise ScenarioValidationError(
                    f"{where}: attribute {name!r} needs at least two categories"
                )
            ref = attr["reference_distribution"]
            if ref != "uniform":
                named = (raw.get("reference_distributions") or {})
                if ref not in named:
                    raise ScenarioValidationError(
                        f"{where}: attribute {name!r} names reference distribution {ref!r}, "
                        "which the file does not define"
                    )
                probs = named[ref].get("probabilities", {})
                missing = [c for c in categories if c not in probs]
                if missing:
                    raise ScenarioValidationError(
                        f"{where}: reference {ref!r} has no mass for categories {missing}"
                    )
                total = sum(float(v) for v in probs.values())
                if abs(total - 1.0) > 1e-6:
                    raise ScenarioValidationError(
                        f"{where}: reference {ref!r} sums to {total:.6f}, not 1.0"
                    )
            if name not in variants:
                raise ScenarioValidationError(
                    f"{where}: attribute {name!r} has no counterfactual_variants block"
                )

        ids: set[str] = set()
        for prompt in raw["base_prompts"]:
            pid = _require(prompt, "id", f"{where}: base_prompts")
            if pid in ids:
                raise ScenarioValidationError(f"{where}: duplicate base prompt id {pid!r}")
            ids.add(pid)
            for key in ("role", "text", "replicates", "actions"):
                _require(prompt, key, f"{where}: base prompt {pid}")
            if int(prompt["replicates"]) < 1:
                raise ScenarioValidationError(
                    f"{where}: base prompt {pid!r} needs replicates >= 1"
                )
            if not prompt["actions"]:
                raise ScenarioValidationError(
                    f"{where}: base prompt {pid!r} needs at least one action"
                )

        for attr_name, entries in variants.items():
            for entry in entries:
                _require(entry, "modifier", f"{where}: {attr_name} variant")
                template = _require(entry, "prompt_template", f"{where}: {attr_name} variant")
                for token in ("{role}", "{action}"):
                    if token not in template:
                        raise ScenarioValidationError(
                            f"{where}: {attr_name} template {template!r} is missing {token}"
                        )

    # -- construction -----------------------------------------------------

    def _build(self, raw: dict[str, Any]) -> Scenario:
        named_refs = raw.get("reference_distributions") or {}
        variants_cfg = raw.get("counterfactual_variants") or {}

        attributes: list[SensitiveAttribute] = []
        for attr in raw["sensitive_attributes"]:
            categories = [str(c) for c in attr["categories"]]
            ref_name = attr["reference_distribution"]
            if ref_name == "uniform":
                reference = ReferenceDistribution(
                    name="uniform",
                    probabilities={c: 1.0 / len(categories) for c in categories},
                    rationale=(
                        "Every category is an equally acceptable default; "
                        "no group is the norm."
                    ),
                    provenance="Normative choice recorded in the scenario file.",
                )
            else:
                spec = named_refs[ref_name]
                reference = ReferenceDistribution(
                    name=ref_name,
                    probabilities={str(k): float(v) for k, v in spec["probabilities"].items()},
                    rationale=str(spec.get("rationale", "")).strip(),
                    provenance=str(spec.get("provenance", "")).strip(),
                )
            attributes.append(
                SensitiveAttribute(
                    name=attr["name"],
                    reference_distribution=reference,
                    classifier=attr["classifier"],
                    categories=categories,
                    counterfactual_modifiers=[
                        e["modifier"] for e in variants_cfg.get(attr["name"], [])
                    ],
                    ambiguous_label=attr.get("ambiguous_label", "ambiguous"),
                )
            )

        base_prompts = [
            BasePrompt(
                id=p["id"],
                role=p["role"],
                text=p["text"],
                replicates=int(p["replicates"]),
                actions=[str(a) for a in p["actions"]],
            )
            for p in raw["base_prompts"]
        ]

        neutral_template = raw.get("neutral_prompt_template", "a {role} {action}")
        expanded = self._expand(base_prompts, attributes, variants_cfg, neutral_template)

        return Scenario(
            id=raw["scenario_id"],
            version=str(raw["version"]),
            domain=raw["domain"],
            sensitive_attributes=attributes,
            base_prompts=base_prompts,
            neutral_prompt_template=neutral_template,
            human_review_fraction=float(
                (raw.get("human_review") or {}).get("sample_fraction", 0.15)
            ),
            source_path=self.config_path,
            _variants=expanded,
        )

    @staticmethod
    def _expand(
        base_prompts: list[BasePrompt],
        attributes: list[SensitiveAttribute],
        variants_cfg: dict[str, Any],
        neutral_template: str,
    ) -> list[PromptVariant]:
        """Pair every role and action with the neutral prompt and each modifier.

        The count this produces is the number the reader is asked to confirm
        before any money is spent: four roles, three actions each, seven
        conditions per action.
        """
        out: list[PromptVariant] = []
        for prompt in base_prompts:
            for action in prompt.actions:
                slug = action.replace(" ", "_")
                out.append(
                    PromptVariant(
                        id=f"{prompt.id}__{slug}__neutral",
                        text=neutral_template.format(role=prompt.role, action=action),
                        role=prompt.id,
                        action=action,
                        condition="neutral",
                        attribute=None,
                        modifier=None,
                        replicates=prompt.replicates,
                    )
                )
                for attribute in attributes:
                    for entry in variants_cfg.get(attribute.name, []):
                        modifier = entry["modifier"]
                        out.append(
                            PromptVariant(
                                id=f"{prompt.id}__{slug}__{attribute.name}_{modifier}",
                                text=entry["prompt_template"].format(
                                    role=prompt.role, action=action
                                ),
                                role=prompt.id,
                                action=action,
                                condition=f"{attribute.name}:{modifier}",
                                attribute=attribute.name,
                                modifier=modifier,
                                replicates=prompt.replicates,
                            )
                        )
        return out


__all__ = [
    "BasePrompt",
    "PromptVariant",
    "ReferenceDistribution",
    "Scenario",
    "ScenarioStore",
    "ScenarioValidationError",
    "SensitiveAttribute",
]
