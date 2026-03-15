"""Counterfactual generation for fairness testing."""

import re
from dataclasses import dataclass

from fairbench.core.types import (
    CounterfactualGroup,
    CounterfactualVariant,
    Scenario,
    SensitiveAttribute,
)
from fairbench.counterfactual.templates import AttributeTemplates


@dataclass
class ExpandedPrompt:
    """A prompt expanded from a scenario, possibly a counterfactual."""

    prompt: str
    scenario_id: str
    is_counterfactual: bool
    attribute: str | None = None
    attribute_value: str | None = None
    original_prompt: str | None = None


class CounterfactualGenerator:
    """Generates counterfactual variants of prompts for fairness testing.

    The generator can:
    1. Expand pre-defined counterfactuals from scenarios
    2. Automatically generate counterfactuals by substituting attributes
    """

    def __init__(self) -> None:
        self.templates = AttributeTemplates()

    def expand_scenario(self, scenario: Scenario) -> list[ExpandedPrompt]:
        """Expand a scenario into all its prompts including counterfactuals.

        Args:
            scenario: The scenario to expand.

        Returns:
            List of expanded prompts (base + all counterfactual variants).
        """
        prompts = []

        # Add the base prompt
        prompts.append(
            ExpandedPrompt(
                prompt=scenario.prompt,
                scenario_id=scenario.id,
                is_counterfactual=False,
            )
        )

        # Add all counterfactual variants
        for cf_group in scenario.counterfactuals:
            for variant in cf_group.variants:
                attr_str = (
                    cf_group.attribute.value
                    if isinstance(cf_group.attribute, SensitiveAttribute)
                    else cf_group.attribute
                )
                prompts.append(
                    ExpandedPrompt(
                        prompt=variant.prompt,
                        scenario_id=scenario.id,
                        is_counterfactual=True,
                        attribute=attr_str,
                        attribute_value=variant.attribute_value,
                        original_prompt=scenario.prompt,
                    )
                )

        return prompts

    def expand_scenarios(self, scenarios: list[Scenario]) -> list[ExpandedPrompt]:
        """Expand multiple scenarios.

        Args:
            scenarios: List of scenarios to expand.

        Returns:
            List of all expanded prompts.
        """
        all_prompts = []
        for scenario in scenarios:
            all_prompts.extend(self.expand_scenario(scenario))
        return all_prompts

    def generate_counterfactual(
        self,
        prompt: str,
        attribute: SensitiveAttribute | str,
        source_value: str,
        target_value: str,
    ) -> str:
        """Generate a counterfactual by substituting attribute terms.

        Args:
            prompt: The original prompt.
            attribute: The sensitive attribute to modify.
            source_value: The current attribute value in the prompt.
            target_value: The desired attribute value.

        Returns:
            The modified prompt with substituted terms.
        """
        source_terms = self.templates.get_terms(attribute, source_value)
        target_terms = self.templates.get_terms(attribute, target_value)

        modified = prompt

        # Handle dictionary of term categories
        if isinstance(source_terms, dict) and isinstance(target_terms, dict):
            for category in source_terms:
                if category in target_terms:
                    source_list = source_terms[category]
                    target_list = target_terms[category]
                    modified = self._substitute_terms(modified, source_list, target_list)
        # Handle simple list of terms
        elif isinstance(source_terms, list) and isinstance(target_terms, list):
            modified = self._substitute_terms(modified, source_terms, target_terms)

        return modified

    def _substitute_terms(
        self, text: str, source_terms: list[str], target_terms: list[str]
    ) -> str:
        """Substitute terms in text.

        Args:
            text: The text to modify.
            source_terms: Terms to replace.
            target_terms: Replacement terms.

        Returns:
            Modified text.
        """
        if not source_terms or not target_terms:
            return text

        modified = text
        for i, source in enumerate(source_terms):
            target = target_terms[min(i, len(target_terms) - 1)]
            # Case-insensitive replacement preserving case
            pattern = re.compile(re.escape(source), re.IGNORECASE)
            modified = pattern.sub(
                lambda m: self._match_case(m.group(), target), modified
            )

        return modified

    def _match_case(self, source: str, target: str) -> str:
        """Match the case of source in the target string.

        Args:
            source: The original string whose case to match.
            target: The target string to transform.

        Returns:
            Target string with case matched to source.
        """
        if source.isupper():
            return target.upper()
        elif source.islower():
            return target.lower()
        elif source[0].isupper():
            return target.capitalize()
        return target

    def generate_name_swap(
        self,
        prompt: str,
        source_name: str,
        target_ethnicity: str,
        target_gender: str | None = None,
    ) -> tuple[str, str]:
        """Generate a counterfactual by swapping names.

        Args:
            prompt: The original prompt containing a name.
            source_name: The name to replace.
            target_ethnicity: The target ethnicity for the new name.
            target_gender: Optional gender for name selection.

        Returns:
            Tuple of (modified prompt, new name used).
        """
        target_names = self.templates.get_names(target_ethnicity, target_gender)
        if not target_names:
            return prompt, source_name

        # Pick the first available name
        target_name = target_names[0]

        # Replace the name, preserving case
        pattern = re.compile(re.escape(source_name), re.IGNORECASE)
        modified = pattern.sub(
            lambda m: self._match_case(m.group(), target_name), prompt
        )

        return modified, target_name

    def create_counterfactual_group(
        self,
        base_prompt: str,
        attribute: SensitiveAttribute | str,
        values: list[str],
        base_value: str | None = None,
    ) -> CounterfactualGroup:
        """Create a counterfactual group from a base prompt.

        Args:
            base_prompt: The original prompt.
            attribute: The sensitive attribute to vary.
            values: The attribute values to create variants for.
            base_value: The attribute value in the base prompt (if any).

        Returns:
            A CounterfactualGroup with generated variants.
        """
        variants = []

        for value in values:
            if base_value:
                # Generate by substitution
                modified = self.generate_counterfactual(
                    base_prompt, attribute, base_value, value
                )
            else:
                # Just note the value, prompt needs manual specification
                modified = base_prompt

            variants.append(
                CounterfactualVariant(
                    prompt=modified,
                    attribute_value=value,
                )
            )

        return CounterfactualGroup(attribute=attribute, variants=variants)
