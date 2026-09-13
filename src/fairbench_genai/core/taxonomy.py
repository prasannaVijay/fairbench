"""Declared category spaces for sensitive attributes.

Three vocabularies for the same demographic axes grew up independently in
this codebase, and nothing reconciled them:

- counterfactual generation names an ethnicity ``chinese`` and a gender
  ``male``;
- the text classifier reports ``east_asian`` and ``masculine``;
- the image classifier reports ``non-binary`` alongside ``male``.

A variant generated as ``chinese`` therefore came back labelled
``east_asian``, and any metric joining a requested attribute to a detected
one was matching across label spaces that shared no strings.

This module declares one canonical space per axis and records every other
spelling as an alias onto it. Detection labels were chosen as canonical
because the metrics count what a classifier emits.

The canonical space is also the source of K for Output Diversity Entropy.
Deriving K from the categories observed in a run lets a category that never
appears shrink the denominator, which raises the normalised entropy and
hides exactly the erasure the metric exists to catch.
"""

from dataclasses import dataclass, field

# Labels that mean "the classifier could not tell". These are never
# categories: they are excluded from K and reported as a coverage shortfall,
# so that an unreliable classifier cannot read as a diverse model.
UNCLASSIFIED = frozenset(
    {"unknown", "ambiguous", "multiple", "not_applicable", "none", ""}
)


@dataclass(frozen=True)
class Taxonomy:
    """A declared set of categories for one sensitive attribute."""

    attribute: str
    categories: tuple[str, ...]
    aliases: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    @property
    def size(self) -> int:
        """K: the number of declared categories, excluding unclassified."""
        return len(self.categories)

    def canonical(self, label: str | None) -> str | None:
        """Resolve a label from any known vocabulary to its canonical form.

        Returns None for an unclassified label or one this taxonomy does not
        recognise, so callers can count coverage rather than silently folding
        unknowns into a category.
        """
        if label is None:
            return None
        key = str(label).strip().lower().replace(" ", "_").replace("-", "_")
        if key in UNCLASSIFIED:
            return None
        if key in self.categories:
            return key
        return self.aliases.get(key)

    def __contains__(self, label: object) -> bool:
        return self.canonical(str(label)) is not None


GENDER = Taxonomy(
    attribute="gender",
    categories=("male", "female", "neutral"),
    aliases={
        # text classifier (pronoun signal)
        "masculine": "male",
        "feminine": "female",
        # counterfactual generation
        "man": "male",
        "woman": "female",
        "boy": "male",
        "girl": "female",
        # image classifier
        "non_binary": "neutral",
        "nonbinary": "neutral",
        "enby": "neutral",
    },
    notes=(
        "The text classifier detects a pronoun signal rather than an identity. "
        "Non-binary identity and gender-neutral pronoun use collapse into "
        "'neutral' and cannot be told apart at Layer 1; report that limitation "
        "alongside any result broken down on this axis."
    ),
)

ETHNICITY = Taxonomy(
    attribute="ethnicity",
    categories=(
        "east_asian",
        "south_asian",
        "hispanic_latino",
        "black_african",
        "middle_eastern",
        "white_western",
    ),
    aliases={
        # counterfactual generation names a narrower instance of each bucket
        "chinese": "east_asian",
        "indian": "south_asian",
        "hispanic": "hispanic_latino",
        "african_american": "black_african",
        "arabic": "middle_eastern",
        "anglo": "white_western",
        # other common spellings
        "latino": "hispanic_latino",
        "latinx": "hispanic_latino",
        "arab": "middle_eastern",
    },
    notes=(
        "Generation names a narrower group than detection can resolve: a "
        "'chinese' name maps onto 'east_asian', which also catches Korean and "
        "Vietnamese surnames. Treat a result on this axis as regional rather "
        "than national. Reference distributions drawn from census categories "
        "do not share this vocabulary and must be restated in it; in "
        "particular a single census 'asian' figure cannot be split across "
        "east_asian and south_asian without a source for the split."
    ),
)

SKIN_TONE = Taxonomy(
    attribute="skin_tone",
    categories=(
        "light",
        "medium_light",
        "medium",
        "medium_dark",
        "dark",
        "very_dark",
    ),
    notes="An ordered scale reported by the image classifier.",
)

AGE = Taxonomy(
    attribute="age",
    categories=("young", "middle_aged", "elderly"),
    aliases={"old": "elderly", "older": "elderly", "senior": "elderly"},
)

RELIGION = Taxonomy(
    attribute="religion",
    categories=(
        "christian",
        "muslim",
        "jewish",
        "hindu",
        "buddhist",
        "atheist",
    ),
    aliases={"islamic": "muslim", "agnostic": "atheist", "secular": "atheist"},
)

TAXONOMIES: dict[str, Taxonomy] = {
    "gender": GENDER,
    "ethnicity": ETHNICITY,
    "race": ETHNICITY,
    "nationality": ETHNICITY,
    "skin_tone": SKIN_TONE,
    "age": AGE,
    "religion": RELIGION,
}

# Which axis each key of EvaluatedOutput.detected_entities belongs to. Text
# and image classifiers use different keys for the same axis, which is why
# counting every key together produced a distribution mixing gender with
# ethnicity.
ENTITY_KEY_TO_ATTRIBUTE: dict[str, str] = {
    "gender_signal": "gender",
    "gender": "gender",
    "name_origins": "ethnicity",
    "skin_tone": "skin_tone",
}


def get_taxonomy(attribute: str | None) -> Taxonomy | None:
    """Look up the declared taxonomy for an attribute, if there is one."""
    if attribute is None:
        return None
    return TAXONOMIES.get(str(attribute).strip().lower())
