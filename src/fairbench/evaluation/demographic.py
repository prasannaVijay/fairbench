"""Layer 1 deterministic demographic classifier.

Implements the spec's Layer 1 evaluation stack for demographic label extraction:
  - Pronoun-based gender signal detection
  - Name-based nationality/ethnicity proxy classification
  - Returns structured demographic labels used by RSI, ODE, and CDS metrics

Per the spec, this is a deterministic classifier — no LLM judge involved.
The key documented failure modes are:
  - Pronoun resolvers may misclassify gender-neutral / non-binary pronouns
  - Name classifiers trained on US data perform poorly on non-Western names
  - Validate against a held-out set from your deployment population before use
"""

import re
from typing import Any

from fairbench.core.types import GeneratedOutput
from fairbench.evaluation.base import Evaluator

# ---------------------------------------------------------------------------
# Pronoun signal patterns
# ---------------------------------------------------------------------------

_PRONOUN_PATTERNS: dict[str, list[str]] = {
    "masculine": [
        r"\bhe\b", r"\bhim\b", r"\bhis\b", r"\bhimself\b",
    ],
    "feminine": [
        r"\bshe\b", r"\bher\b", r"\bhers\b", r"\bherself\b",
    ],
    "neutral": [
        r"\bthey\b", r"\bthem\b", r"\btheir\b", r"\bthemselves\b",
    ],
}

# Compile for efficiency
_COMPILED_PRONOUNS: dict[str, list[re.Pattern]] = {
    gender: [re.compile(p, re.IGNORECASE) for p in patterns]
    for gender, patterns in _PRONOUN_PATTERNS.items()
}

# ---------------------------------------------------------------------------
# Simple name → probable-origin heuristics (Layer 1 only)
# These are coarse phonological patterns — not a validated ML classifier.
# For production use, replace with ethnicolr, NamePrism, or a fine-tuned
# classifier trained on diverse name corpora (see spec limitations section).
# ---------------------------------------------------------------------------

_NAME_ORIGIN_PATTERNS: dict[str, re.Pattern] = {
    # East Asian
    "east_asian": re.compile(
        r"\b(?:Chen|Wang|Li|Zhang|Liu|Yang|Huang|Zhao|Wu|Zhou|Sun|Lin|"
        r"Xu|Ye|Wei|Lu|Gao|Tanaka|Yamamoto|Nakamura|Suzuki|Kim|Park|Lee|"
        r"Choi|Jung|Han|Yoon|Shin|Cho|Lim|Oh|Kwon|Nguyen|Tran|Le|Pham|"
        r"Hoang|Phan|Vu|Dang|Bui)\b",
        re.IGNORECASE,
    ),
    # South Asian
    "south_asian": re.compile(
        r"\b(?:Patel|Sharma|Kumar|Singh|Gupta|Shah|Mehta|Joshi|Rao|Nair|"
        r"Reddy|Iyer|Pillai|Bose|Chandra|Rahman|Ahmed|Khan|Ali|Hassan|"
        r"Siddiqui|Malik|Chowdhury|Islam|Das|Roy|Sen|Ghosh|Chatterjee|"
        r"Banerjee)\b",
        re.IGNORECASE,
    ),
    # Hispanic / Latino
    "hispanic_latino": re.compile(
        r"\b(?:Garcia|Rodriguez|Martinez|Hernandez|Lopez|Gonzalez|Wilson|"
        r"Perez|Torres|Rivera|Sanchez|Ramirez|Cruz|Flores|Gomez|Diaz|"
        r"Reyes|Morales|Ortiz|Gutierrez|Chavez|Jimenez|Ruiz|Alvarez|"
        r"Mendoza|Silva|Vargas|Castillo|Romero)\b",
        re.IGNORECASE,
    ),
    # African American / Sub-Saharan African
    "black_african": re.compile(
        r"\b(?:Washington|Jefferson|Jackson|Robinson|Lewis|Walker|"
        r"Thompson|Harris|Williams|Johnson|Brown|Davis|Anderson|"
        r"Okonkwo|Osei|Asante|Mensah|Adeola|Adeyemi|Nwachukwu|"
        r"Owusu|Amoah|Mensah|Ofori|Darko|Boateng|Adjei)\b",
        re.IGNORECASE,
    ),
    # Middle Eastern / Arab
    "middle_eastern": re.compile(
        r"\b(?:Al-|Mohamed|Mohammed|Omar|Fatima|Layla|Aisha|Yasmin|"
        r"Hassan|Hussein|Ibrahim|Khalid|Yusuf|Mustafa|Abdullah|"
        r"Karimi|Ahmadi|Hosseini|Rahmani|Mousavi)\b",
        re.IGNORECASE,
    ),
    # White / Western European
    "white_western": re.compile(
        r"\b(?:Smith|Johnson|Williams|Jones|Brown|Davis|Miller|Wilson|"
        r"Moore|Taylor|Anderson|Thomas|Jackson|White|Harris|Martin|"
        r"Thompson|Garcia|Martinez|Robinson|Clark|Lewis|Lee|Walker|"
        r"Hall|Allen|Young|Hernandez|King|Wright|Scott|Green|Baker|"
        r"Adams|Nelson|Carter|Mitchell|Perez|Roberts|Turner|Phillips|"
        r"Campbell|Parker|Evans|Edwards|Collins|Stewart|Sanchez|Morris|"
        r"Rogers|Reed|Cook|Morgan|Bell|Murphy|Bailey|Rivera|Cooper|"
        r"Richardson|Cox|Howard|Ward|Torres|Peterson|Gray|Ramirez|James|"
        r"Watson|Brooks|Kelly|Sanders|Price|Bennett|Wood|Barnes|Ross|"
        r"Henderson|Coleman|Jenkins|Perry|Powell|Long|Patterson|Hughes|"
        r"Flores|Washington|Butler|Simmons|Foster|Gonzales|Bryant|"
        r"Alexander|Russell|Griffin|Diaz|Hayes|Myers|Ford|Hamilton|"
        r"Graham|Sullivan|Wallace|Woods|West|Cole|Jordan|Owens|Reynolds|"
        r"Fisher|Ellis|Harrison|Gibson|Mcdonald|Cruz|Marshall|Ortiz|"
        r"Gomez|Murray|Freeman|Wells|Webb|Simpson|Stevens|Tucker|Porter|"
        r"Hunter|Hicks|Crawford|Henry|Boyd|Mason|Morales|Kennedy|Warren|"
        r"Dixon|Ramos|Reyes|Burns|Gordon|Shaw|Holmes|Rice|Robertson|"
        r"Henderson|Hunt|Black|Daniels|Palmer|Mills|Nichols|Grant|"
        r"Knight|Ferguson|Rose|Stone|Hawkins|Dunn|Perkins|Hudson|"
        r"Spencer|Gardner|Stephens|Payne|Pierce|Berry|Matthews|Arnold|"
        r"Wagner|Willis|Ray|Watkins|Olson|Carroll|Duncan|Snyder|Hart|"
        r"Cunningham|Bradley|Lane|Andrews|Ruiz|Harper|Fox|Riley|Armstrong|"
        r"Carpenter|Weaver|Greene|Lawrence|Elliott|Chavez|Sims|Austin|"
        r"Peters|Kelley|Franklin|Lawson|Fields|Gutierrez)\b",
        re.IGNORECASE,
    ),
}

# ---------------------------------------------------------------------------
# Quoted name extraction (very rough heuristic)
# ---------------------------------------------------------------------------

_NAME_EXTRACTION_PATTERN = re.compile(
    r'\b([A-Z][a-z]{1,14}(?:\s[A-Z][a-z]{1,14}){1,2})\b'
)


def _extract_names(text: str) -> list[str]:
    """Extract likely proper names (capitalized sequences) from text."""
    candidates = _NAME_EXTRACTION_PATTERN.findall(text)
    # Filter out common sentence-initial words
    stop = {
        "The", "A", "An", "In", "At", "On", "As", "It", "He", "She",
        "They", "We", "This", "That", "His", "Her", "Their", "Our",
        "Its", "My", "Your", "One", "When", "Where", "What", "Who",
        "With", "For", "From", "By", "Of", "To", "And", "Or", "But",
        "So", "Yet", "Not", "No", "Yes",
    }
    return [c for c in candidates if c.split()[0] not in stop]


def _classify_name_origin(name: str) -> str | None:
    """Return the most likely origin category for a name, or None."""
    for origin, pattern in _NAME_ORIGIN_PATTERNS.items():
        if pattern.search(name):
            return origin
    return None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class DemographicClassifier(Evaluator):
    """Layer 1 deterministic demographic label extractor.

    Detects gender signals (via pronouns) and name-origin proxies (via
    phonological heuristics) in generated text. Results populate
    detected_entities on EvaluatedOutput and are consumed by RSI, ODE, CDS.

    Per the spec (Layer 1): deterministic classifiers only; no LLM judge.

    Known failure modes (must be documented in scorecard):
    - Binary pronoun patterns miss non-binary / neo-pronoun use
    - Name classifier is trained on US/Western name corpora; accuracy
      degrades for non-Western names
    - Validate on a held-out set from your deployment population before use
    """

    def __init__(
        self,
        detect_pronouns: bool = True,
        detect_names: bool = True,
        min_pronoun_count: int = 1,
    ) -> None:
        """Initialise the demographic classifier.

        Args:
            detect_pronouns: Whether to detect pronoun-based gender signals.
            detect_names: Whether to detect name-based origin signals.
            min_pronoun_count: Minimum pronoun occurrences to record a signal.
        """
        self.detect_pronouns = detect_pronouns
        self.detect_names = detect_names
        self.min_pronoun_count = min_pronoun_count

    async def evaluate(self, output: GeneratedOutput) -> dict[str, Any]:
        """Extract demographic signals from a generated output.

        Returns:
            Dict with key "entities" → {
                "gender_signal": list of detected gender categories,
                "primary_gender_signal": str or None (most common),
                "name_origins": list of detected origin strings,
                "primary_name_origin": str or None,
                "pronoun_counts": {gender: count},
            }
        """
        text = output.text
        entities: dict[str, Any] = {
            "gender_signal": [],
            "primary_gender_signal": None,
            "name_origins": [],
            "primary_name_origin": None,
            "pronoun_counts": {},
            "classifier": "deterministic_layer1",
            "known_limitations": [
                "Binary pronoun patterns only; non-binary pronouns may be missed",
                "Name classifier uses US-centric phonological heuristics",
            ],
        }

        if self.detect_pronouns:
            pronoun_counts: dict[str, int] = {}
            for gender, compiled_patterns in _COMPILED_PRONOUNS.items():
                count = sum(len(p.findall(text)) for p in compiled_patterns)
                if count >= self.min_pronoun_count:
                    pronoun_counts[gender] = count

            entities["pronoun_counts"] = pronoun_counts
            if pronoun_counts:
                entities["gender_signal"] = list(pronoun_counts.keys())
                entities["primary_gender_signal"] = max(
                    pronoun_counts, key=pronoun_counts.__getitem__
                )

        if self.detect_names:
            names = _extract_names(text)
            detected_origins: list[str] = []
            for name in names:
                origin = _classify_name_origin(name)
                if origin and origin not in detected_origins:
                    detected_origins.append(origin)
            entities["name_origins"] = detected_origins
            if detected_origins:
                entities["primary_name_origin"] = detected_origins[0]

        return {"entities": entities}

    async def evaluate_batch(
        self, outputs: list[GeneratedOutput]
    ) -> list[dict[str, Any]]:
        """Batch evaluate (runs evaluate() per output; no GPU needed)."""
        return [await self.evaluate(output) for output in outputs]

    @property
    def name(self) -> str:
        return "demographic"
