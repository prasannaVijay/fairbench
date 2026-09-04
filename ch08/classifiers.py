"""Deterministic demographic classifiers for the Chapter 8 pilot.

Two classifiers run in sequence on every generated image, one for apparent
gender and one for Fitzpatrick skin tone. They are first-layer evaluators in the
sense of Chapter 7: deterministic, independently validatable, and not a model
judging another model.

Both read their scores from the metadata sidecar written next to each image.
In a live run that sidecar is produced by the classifier binary named in the
scenario file; on the recorded-output path it is produced by the fixture replay.
Either way the class here owns the part that decides the label, which is the
confidence threshold, so that changing the threshold genuinely changes the run.

A word on what these labels are. The gender classifier reads apparent gender
presentation in an image, which is not gender, and the Fitzpatrick classifier
reads apparent tone against a scale built to describe how skin responds to
ultraviolet light rather than to categorise people. Both are proxies, both are
imprecise at the edges, and the ``ambiguous`` label exists so that the
imprecision is counted rather than guessed at.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ClassifierError(RuntimeError):
    """Raised when an image carries no scores the classifier can read."""


@dataclass(frozen=True)
class ClassificationResult:
    """One classifier's read of one image."""

    label: str
    confidence: float
    model: str
    is_ambiguous: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "model": self.model,
            "is_ambiguous": self.is_ambiguous,
        }


def sidecar_path(image_path: str | Path) -> Path:
    """The metadata file that travels alongside an image."""
    p = Path(image_path)
    return p.with_suffix(".json")


def load_metadata(image_path: str | Path) -> dict[str, Any]:
    """Read the metadata attached to an image at generation time.

    The metadata is what makes a metric traceable: it names the prompt variant,
    the role, the action, the condition, and the replicate index behind the
    image, so a score can always be walked back to the request that produced it.
    """
    path = sidecar_path(image_path)
    if not path.exists():
        raise ClassifierError(f"no metadata sidecar for {image_path}")
    return json.loads(path.read_text(encoding="utf-8"))


class _ThresholdedClassifier:
    """Shared behaviour: read the recorded score, then apply the threshold.

    Any output whose confidence falls below ``confidence_threshold`` is labelled
    ambiguous. Ambiguous outputs are not discarded. They are counted, reported,
    and kept out of the distributions, because an uncertain read is not evidence
    about representation and a rising ambiguous rate is a signal about the
    classifier rather than about the model.
    """

    attribute = ""
    ambiguous_label = "ambiguous"

    def __init__(self, model: str, confidence_threshold: float) -> None:
        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must fall in (0, 1]")
        self.model = model
        self.confidence_threshold = float(confidence_threshold)

    def classify(self, image_path: str | Path) -> ClassificationResult:
        metadata = load_metadata(image_path)
        scores = (metadata.get("classifier_scores") or {}).get(self.attribute)
        if not scores:
            raise ClassifierError(
                f"{self.attribute} scores missing from the metadata for {image_path}"
            )
        label = str(scores["label"])
        confidence = float(scores["confidence"])
        if confidence < self.confidence_threshold:
            return ClassificationResult(
                label=self.ambiguous_label,
                confidence=confidence,
                model=self.model,
                is_ambiguous=True,
            )
        return ClassificationResult(
            label=label, confidence=confidence, model=self.model, is_ambiguous=False
        )


class GenderClassifier(_ThresholdedClassifier):
    """Apparent-gender classifier. Categories: male, female, non_binary."""

    attribute = "gender"


class FitzpatrickClassifier(_ThresholdedClassifier):
    """Fitzpatrick skin-tone classifier. Categories: I through VI.

    The scale's known imprecision for darker tones is a documented limitation of
    every score this class produces, and it belongs in the run's limitation
    record rather than in a footnote.
    """

    attribute = "skin_tone"


__all__ = [
    "ClassificationResult",
    "ClassifierError",
    "FitzpatrickClassifier",
    "GenderClassifier",
    "load_metadata",
    "sidecar_path",
]
