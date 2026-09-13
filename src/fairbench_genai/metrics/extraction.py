"""Shared category extraction for the distributional metrics.

RSI, ODE and SAR all ask the same question of a run: which demographic
category does each output belong to? They used to answer it by counting
``counterfactual_value``, which is the variant that was *requested*, set once
at generation time and never updated from anything the evaluator detected.
Because the counterfactual generator emits a balanced variant set by
construction, that distribution is close to uniform whatever model is under
test, so the metrics described the prompt design rather than the model.

This module reads the detected attribute instead, resolves every label through
a declared taxonomy so that generation and detection vocabularies meet, and
reports how much of the run could be classified at all.

One axis at a time. A run can carry both a gender signal and a name-origin
signal, and counting them together produces a distribution that mixes the two.
When more than one axis is present the caller has to say which one it wants.
"""

from collections import Counter
from dataclasses import dataclass

from fairbench_genai.core.exceptions import MetricError
from fairbench_genai.core.taxonomy import (
    ENTITY_KEY_TO_ATTRIBUTE,
    Taxonomy,
    get_taxonomy,
)
from fairbench_genai.core.types import EvaluatedOutput

#: Read the category the evaluator detected in the output.
DETECTED = "detected"
#: Read the counterfactual variant that was requested when generating.
REQUESTED = "requested"


@dataclass(frozen=True)
class ExtractedCategories:
    """Canonical category labels for one attribute axis of a run."""

    attribute: str | None
    source: str
    labels: list[str]
    n_outputs: int
    taxonomy: Taxonomy | None

    @property
    def n_classified(self) -> int:
        return len(self.labels)

    @property
    def coverage(self) -> float:
        """Share of outputs that could be placed in a category."""
        if self.n_outputs == 0:
            return 0.0
        return self.n_classified / self.n_outputs

    @property
    def k_source(self) -> str:
        return "declared" if self.taxonomy is not None else "observed"

    @property
    def category_space(self) -> tuple[str, ...]:
        """The categories the run could have produced.

        Declared by the taxonomy where one exists. Falling back to the
        observed categories lets an absent category shrink the space, which
        is why the fallback is recorded in ``k_source``.
        """
        if self.taxonomy is not None:
            return self.taxonomy.categories
        return tuple(sorted(set(self.labels)))

    @property
    def k(self) -> int:
        return len(self.category_space)

    def counts(self) -> dict[str, int]:
        """Counts across the whole category space, including absent ones."""
        counted = Counter(self.labels)
        return {category: counted.get(category, 0) for category in self.category_space}

    def distribution(self) -> dict[str, float]:
        total = self.n_classified
        if total == 0:
            return {category: 0.0 for category in self.category_space}
        return {category: count / total for category, count in self.counts().items()}

    def as_details(self) -> dict:
        """Provenance to attach to a metric result."""
        return {
            "attribute": self.attribute,
            "category_source": self.source,
            "category_space": list(self.category_space),
            "k": self.k,
            "k_source": self.k_source,
            "n_classified": self.n_classified,
            "n_outputs": self.n_outputs,
            "classification_coverage": self.coverage,
        }


def available_attributes(
    outputs: list[EvaluatedOutput], source: str = DETECTED
) -> set[str]:
    """Which attribute axes this run carries."""
    found: set[str] = set()
    for output in outputs:
        if source == DETECTED:
            for key, values in output.detected_entities.items():
                attribute = ENTITY_KEY_TO_ATTRIBUTE.get(key)
                if attribute and values:
                    found.add(attribute)
        else:
            if output.counterfactual_attribute and output.counterfactual_value:
                found.add(str(output.counterfactual_attribute).lower())
    return found


def axes_to_score(
    outputs: list[EvaluatedOutput], attribute: str | None, source: str = DETECTED
) -> list[str]:
    """Which axes a metric should score when the caller named none.

    A run can carry several axes at once; an image run reports both a gender
    signal and a skin tone. Scoring them together would mix distributions, and
    picking one silently would hide the rest, so every axis present is scored
    and the caller decides what to do with the set.
    """
    if attribute is not None:
        return [str(attribute).lower()]
    found = available_attributes(outputs, source)
    if not found:
        raise MetricError(
            f"No {source} demographic categories were found in these outputs. "
            "Run the demographic evaluator, or pass source='requested' to "
            "score the counterfactual variants that were asked for."
        )
    return sorted(found)


def _resolve_attribute(
    outputs: list[EvaluatedOutput], attribute: str | None, source: str
) -> str:
    if attribute is not None:
        return str(attribute).lower()
    found = available_attributes(outputs, source)
    if not found:
        raise MetricError(
            f"No {source} demographic categories were found in these outputs. "
            "Run the demographic evaluator, or pass source='requested' to "
            "score the counterfactual variants that were asked for."
        )
    if len(found) > 1:
        listed = ", ".join(sorted(found))
        raise MetricError(
            f"These outputs carry more than one attribute axis ({listed}). "
            "Counting them together mixes distributions, so pass "
            "attribute='<one of them>' to choose."
        )
    return found.pop()


def _labels_for(output: EvaluatedOutput, attribute: str, source: str) -> list[str]:
    if source == REQUESTED:
        if (
            output.counterfactual_value
            and str(output.counterfactual_attribute).lower() == attribute
        ):
            return [output.counterfactual_value]
        return []
    found: list[str] = []
    for key, values in output.detected_entities.items():
        if ENTITY_KEY_TO_ATTRIBUTE.get(key) != attribute:
            continue
        found.extend(str(value) for value in values)
    return found


def extract_categories(
    outputs: list[EvaluatedOutput],
    attribute: str | None = None,
    source: str = DETECTED,
) -> ExtractedCategories:
    """Resolve one canonical category per output for a single attribute axis.

    Args:
        outputs: The evaluated outputs of a run.
        attribute: Which axis to read. Inferred when the run carries exactly
            one, and required when it carries more than one.
        source: ``detected`` reads what the evaluator found in the output;
            ``requested`` reads the counterfactual variant that was asked for.

    Returns:
        The canonical labels, the category space they sit in, and how much of
        the run could be classified.

    An output that carries several labels on one axis contributes its first,
    which matches the ``primary_`` signals the demographic classifier reports.
    An output whose label is unclassified or outside the taxonomy contributes
    nothing and lowers coverage instead.
    """
    if source not in (DETECTED, REQUESTED):
        raise MetricError(
            f"Unknown category source: {source}. Use '{DETECTED}' or " f"'{REQUESTED}'."
        )

    resolved = _resolve_attribute(outputs, attribute, source)
    taxonomy = get_taxonomy(resolved)

    labels: list[str] = []
    for output in outputs:
        for raw in _labels_for(output, resolved, source):
            label = taxonomy.canonical(raw) if taxonomy is not None else str(raw)
            if label:
                labels.append(label)
                break

    return ExtractedCategories(
        attribute=resolved,
        source=source,
        labels=labels,
        n_outputs=len(outputs),
        taxonomy=taxonomy,
    )
