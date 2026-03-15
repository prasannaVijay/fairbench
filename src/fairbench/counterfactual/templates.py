"""Attribute templates for counterfactual generation."""

from fairbench.core.types import SensitiveAttribute


class AttributeTemplates:
    """Templates for generating counterfactual variants by swapping attributes."""

    # Gender-related terms
    GENDER_TERMS: dict[str, dict[str, list[str]]] = {
        "male": {
            "pronouns_subject": ["he"],
            "pronouns_object": ["him"],
            "pronouns_possessive": ["his"],
            "titles": ["Mr.", "Sir"],
            "relations": ["father", "son", "brother", "husband", "uncle", "nephew", "grandfather"],
            "generic": ["man", "boy", "male", "gentleman"],
        },
        "female": {
            "pronouns_subject": ["she"],
            "pronouns_object": ["her"],
            "pronouns_possessive": ["her", "hers"],
            "titles": ["Ms.", "Mrs.", "Miss", "Ma'am"],
            "relations": ["mother", "daughter", "sister", "wife", "aunt", "niece", "grandmother"],
            "generic": ["woman", "girl", "female", "lady"],
        },
        "neutral": {
            "pronouns_subject": ["they"],
            "pronouns_object": ["them"],
            "pronouns_possessive": ["their", "theirs"],
            "titles": ["Mx."],
            "relations": ["parent", "child", "sibling", "spouse", "pibling", "nibling", "grandparent"],
            "generic": ["person", "individual", "human"],
        },
    }

    # Common names by perceived ethnicity (for name-swapping counterfactuals)
    NAMES_BY_ETHNICITY: dict[str, dict[str, list[str]]] = {
        "anglo": {
            "male": ["James", "John", "Michael", "David", "Robert", "William"],
            "female": ["Emily", "Sarah", "Jessica", "Ashley", "Jennifer", "Elizabeth"],
        },
        "african_american": {
            "male": ["Jamal", "DeShawn", "Terrell", "Marcus", "Andre", "Malik"],
            "female": ["Latoya", "Keisha", "Tanisha", "Aaliyah", "Imani", "Jasmine"],
        },
        "hispanic": {
            "male": ["Jose", "Carlos", "Miguel", "Juan", "Luis", "Diego"],
            "female": ["Maria", "Sofia", "Isabella", "Guadalupe", "Carmen", "Rosa"],
        },
        "chinese": {
            "male": ["Wei", "Jian", "Ming", "Yong", "Hui", "Feng"],
            "female": ["Li", "Mei", "Xia", "Ying", "Jing", "Fang"],
        },
        "indian": {
            "male": ["Raj", "Vikram", "Amit", "Suresh", "Arun", "Sanjay"],
            "female": ["Priya", "Anita", "Sunita", "Deepa", "Lakshmi", "Kavita"],
        },
        "arabic": {
            "male": ["Mohammed", "Ahmed", "Omar", "Hassan", "Ali", "Yusuf"],
            "female": ["Fatima", "Aisha", "Maryam", "Layla", "Noor", "Amira"],
        },
    }

    # Age-related terms
    AGE_TERMS: dict[str, list[str]] = {
        "young": ["young", "youthful", "junior", "early-career"],
        "middle_aged": ["middle-aged", "experienced", "seasoned", "mid-career"],
        "elderly": ["elderly", "senior", "retired", "older", "aging"],
    }

    # Religion-related terms
    RELIGION_TERMS: dict[str, list[str]] = {
        "christian": ["Christian", "Catholic", "Protestant", "church", "Bible"],
        "muslim": ["Muslim", "Islamic", "mosque", "Quran"],
        "jewish": ["Jewish", "Judaism", "synagogue", "Torah"],
        "hindu": ["Hindu", "Hinduism", "temple", "Vedas"],
        "buddhist": ["Buddhist", "Buddhism", "meditation", "dharma"],
        "atheist": ["atheist", "secular", "non-religious", "agnostic"],
    }

    @classmethod
    def get_terms(
        cls, attribute: SensitiveAttribute | str, value: str
    ) -> dict[str, list[str]] | list[str]:
        """Get terms for a specific attribute value.

        Args:
            attribute: The sensitive attribute type.
            value: The specific value (e.g., "male", "anglo").

        Returns:
            Dictionary or list of relevant terms.
        """
        attr_str = attribute.value if isinstance(attribute, SensitiveAttribute) else attribute

        if attr_str == "gender":
            return cls.GENDER_TERMS.get(value, {})
        elif attr_str in ("race", "ethnicity", "nationality"):
            return cls.NAMES_BY_ETHNICITY.get(value, {})
        elif attr_str == "age":
            return cls.AGE_TERMS.get(value, [])
        elif attr_str == "religion":
            return cls.RELIGION_TERMS.get(value, [])

        return []

    @classmethod
    def get_names(cls, ethnicity: str, gender: str | None = None) -> list[str]:
        """Get names for an ethnicity and optional gender.

        Args:
            ethnicity: The ethnicity key.
            gender: Optional gender filter ("male" or "female").

        Returns:
            List of names.
        """
        names_dict = cls.NAMES_BY_ETHNICITY.get(ethnicity, {})
        if gender:
            return names_dict.get(gender, [])
        # Return all names
        all_names = []
        for name_list in names_dict.values():
            all_names.extend(name_list)
        return all_names
