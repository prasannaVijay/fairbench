"""Type definitions for image generation benchmarking."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from fairbench.core.types import (
    EvaluatedOutput,
    GenerationConfig,
    MetricResult,
    ModelInfo,
    RunStatus,
    SentimentScores,
    ToxicityScores,
    _utc_now,
)


class ImageGenerationConfig(BaseModel):
    """Configuration for image generation models."""

    size: str = "1024x1024"
    quality: str = "auto"      # gpt-image-1: "low"|"medium"|"high"|"auto"
    style: str | None = None
    n: int = 1
    seed: int | None = None
    guidance_scale: float | None = None  # SD-specific
    num_inference_steps: int | None = None  # SD-specific
    negative_prompt: str | None = None  # SD-specific

    model_config = {"frozen": True}


class GeneratedImage(BaseModel):
    """Output from an image generation model."""

    prompt: str
    model_info: ModelInfo
    generation_config: ImageGenerationConfig
    image_data: bytes | None = None   # Raw PNG/JPEG bytes
    image_url: str | None = None      # Remote URL (e.g., DALL-E response)
    image_path: Path | None = None    # Local saved path
    revised_prompt: str | None = None  # DALL-E may rewrite the prompt
    latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utc_now)

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    def has_image(self) -> bool:
        """Return True if image content is available (bytes, URL, or path)."""
        return bool(self.image_data or self.image_url or self.image_path)


class ImageAnalysis(BaseModel):
    """Structured demographic and quality analysis of a generated image.

    Produced by VisionAnalyzer (Claude Vision) and/or CLIPEvaluator.
    """

    # Demographic signals
    perceived_gender: str = "unknown"
    # "male" | "female" | "non-binary" | "ambiguous" | "multiple" | "unknown"

    skin_tone_fitzpatrick: int | None = None  # 1 (lightest) – 6 (darkest)
    skin_tone_label: str = "unknown"
    # "light" | "medium-light" | "medium" | "medium-dark" | "dark" | "very-dark" | "unknown"

    # Setting and production quality signals
    setting: str = "unknown"
    # "professional_stadium" | "national_league" | "local_field" | "park"
    # | "studio" | "street" | "generic" | "unknown"

    equipment_quality: str = "unknown"
    # "professional_kit" | "semi_professional" | "amateur" | "generic" | "not_visible" | "unknown"

    pose_type: str = "unknown"
    # "action_competitive" | "action_recreational" | "posed_professional"
    # | "posed_casual" | "portrait" | "unknown"

    age_presentation: str = "unknown"
    # "youth" | "young_adult" | "adult" | "veteran" | "unknown"

    # Quality and bias signals
    image_quality_score: float | None = None  # 1.0 – 10.0
    stereotype_severity: float = 0.0          # 0.0 – 1.0
    stereotypes_detected: list[str] = Field(default_factory=list)

    # Narrative
    description: str = ""
    analysis_confidence: float = 1.0
    analysis_source: str = "unknown"  # "claude_vision" | "clip" | "combined"

    model_config = {"frozen": True}


class EvaluatedImage(BaseModel):
    """A generated image with analysis annotations.

    This is the image-pipeline analogue of EvaluatedOutput.
    Call to_evaluated_output() to bridge into the existing metric infrastructure.
    """

    id: UUID = Field(default_factory=uuid4)
    image: GeneratedImage
    scenario_id: str
    is_counterfactual: bool = False
    counterfactual_attribute: str | None = None
    counterfactual_value: str | None = None
    original_prompt: str | None = None

    # Analysis results
    clip_embedding: list[float] | None = None
    clip_similarities: dict[str, float] = Field(default_factory=dict)
    vision_analysis: ImageAnalysis | None = None

    # Generation-level service signals
    is_refused: bool = False       # True if model refused (NSFW block, etc.)
    refusal_reason: str | None = None

    model_config = {"frozen": True}

    def to_evaluated_output(self) -> EvaluatedOutput:
        """Bridge this evaluated image into the existing metric infrastructure.

        Mapping strategy:
        - embedding       → CLIP visual embedding (for CDS cosine distance, ODE diversity)
        - detected_entities["gender"]     → [perceived_gender]   (for RSI "detected" mode)
        - detected_entities["skin_tone"]  → [skin_tone_label]    (for RSI "detected" mode)
        - detected_entities["setting"]    → [setting]            (for DSI/SAR quality proxy)
        - toxicity.identity_attack        → stereotype_severity   (for HSI)
        - helpfulness_score               → image_quality_score   (for DSI)
        - is_refusal                      → is_refused            (for DSI refusal rate)
        - custom_evaluations              → full analysis dicts
        """
        analysis = self.vision_analysis

        # Build detected_entities for RSI "detected" mode
        detected: dict[str, list[str]] = {}
        if analysis:
            if analysis.perceived_gender != "unknown":
                detected["gender"] = [analysis.perceived_gender]
            if analysis.skin_tone_label != "unknown":
                detected["skin_tone"] = [analysis.skin_tone_label]
            if analysis.setting != "unknown":
                detected["setting"] = [analysis.setting]
            if analysis.equipment_quality != "unknown":
                detected["equipment_quality"] = [analysis.equipment_quality]

        # Map stereotype_severity → ToxicityScores.identity_attack for HSI
        toxicity = None
        if analysis and analysis.stereotype_severity > 0:
            toxicity = ToxicityScores(
                toxicity=analysis.stereotype_severity * 0.5,
                identity_attack=analysis.stereotype_severity,
                insult=analysis.stereotype_severity * 0.3,
            )

        # Map image_quality_score to helpfulness proxy for DSI (0–1 normalized)
        helpfulness: float | None = None
        if analysis and analysis.image_quality_score is not None:
            helpfulness = analysis.image_quality_score / 10.0

        # Build a placeholder text output so EvaluatedOutput validates
        from fairbench.core.types import GeneratedOutput

        placeholder_output = GeneratedOutput(
            text=analysis.description if analysis else f"[image: {self.image.prompt}]",
            prompt=self.image.prompt,
            model_info=self.image.model_info,
            generation_config=GenerationConfig(),
            latency_ms=self.image.latency_ms,
            metadata={"image_url": self.image.image_url, "modality": "image"},
        )

        custom: dict[str, Any] = {
            "modality": "image",
        }
        if analysis:
            custom["vision_analysis"] = analysis.model_dump()
        if self.clip_similarities:
            custom["clip_similarities"] = self.clip_similarities

        return EvaluatedOutput(
            id=self.id,
            output=placeholder_output,
            scenario_id=self.scenario_id,
            is_counterfactual=self.is_counterfactual,
            counterfactual_attribute=self.counterfactual_attribute,
            counterfactual_value=self.counterfactual_value,
            original_prompt=self.original_prompt,
            embedding=self.clip_embedding,
            toxicity=toxicity,
            sentiment=SentimentScores(positive=0.5, negative=0.0, neutral=0.5),
            detected_entities=detected,
            is_refusal=self.is_refused,
            helpfulness_score=helpfulness,
            custom_evaluations=custom,
        )


class ImageEvaluationRun(BaseModel):
    """A complete image benchmark evaluation run."""

    id: UUID = Field(default_factory=uuid4)
    status: RunStatus = RunStatus.PENDING
    model_info: ModelInfo
    scenario_sets: list[str]
    metrics_requested: list[str]

    evaluated_images: list[EvaluatedImage] = Field(default_factory=list)
    metric_results: list[MetricResult] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=_utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None

    def total_images(self) -> int:
        return len(self.evaluated_images)

    def refused_count(self) -> int:
        return sum(1 for img in self.evaluated_images if img.is_refused)
