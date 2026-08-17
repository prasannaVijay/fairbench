"""StubImageAdapter: an offline, deterministic image-generation adapter.

This adapter lets the whole image benchmarking pipeline run end to end without
an API key or a paid model call. It does not draw anything; instead it "replays"
a deterministic, recorded result for each prompt and bakes the demographic
signal a real vision model would later detect into the image metadata, under
``metadata["stub_analysis"]``. The companion StubVisionAnalyzer and
StubCLIPEvaluator (in ``fairbench_genai.evaluation.image.stub``) read that
signal back out, so every layer of the architecture from Chapter 5 is exercised
with real objects and real control flow, just without the network.

The recorded distribution is intentionally skewed to reproduce the book's
running example: when a prompt names no gender, the stub defaults to a male
depiction roughly 80% of the time, so the Representation Skew Index and the
scorecard show a clear, explainable signal. The numbers are illustrative.

Usage:
    from fairbench_genai.adapters.image.stub import StubImageAdapter
    adapter = StubImageAdapter()          # no API key required
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any

from fairbench_genai.adapters.image.base import ImageModelAdapter
from fairbench_genai.core.image_types import GeneratedImage, ImageGenerationConfig
from fairbench_genai.core.types import ModelInfo

# A minimal valid 1x1 transparent PNG. The pipeline only needs bytes that make
# GeneratedImage.has_image() true; nothing decodes this in the stub path.
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class StubImageAdapter(ImageModelAdapter):
    """Deterministic, offline stand-in for a hosted image model.

    Args:
        seed_salt: Salt mixed into the per-prompt hash so a deployment can vary
            the recorded outputs reproducibly.
        female_share_unconstrained: Fraction of *unconstrained* prompts (those
            that name no gender) depicted as female. The remainder are male.
            Defaults to 0.20 to reproduce the book's default-male skew.
        refuse_nonbinary: If True, prompts that explicitly request a non-binary
            subject are "refused" with a content-policy reason, exercising the
            refusal / execution-denominator path. This mirrors a real and
            fairness-relevant provider behaviour rather than endorsing it.
    """

    def __init__(
        self,
        seed_salt: str = "fairbench-stub",
        female_share_unconstrained: float = 0.20,
        refuse_nonbinary: bool = True,
    ) -> None:
        self.seed_salt = seed_salt
        self.female_share_unconstrained = female_share_unconstrained
        self.refuse_nonbinary = refuse_nonbinary

    @property
    def name(self) -> str:
        return "stub-image"

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name="stub-image-1",
            provider="stub",
            version="2026-01",
            parameters={"deterministic": True, "seed_salt": self.seed_salt},
        )

    def _hash(self, prompt: str) -> int:
        digest = hashlib.sha256(f"{self.seed_salt}:{prompt}".encode()).hexdigest()
        return int(digest, 16)

    def _recorded_analysis(self, prompt: str) -> dict[str, Any]:
        """Derive a deterministic, baked-in demographic reading for a prompt."""
        low = prompt.lower()
        h = self._hash(prompt)

        # Gender: honour an explicit request; otherwise apply the default skew.
        if "female" in low:
            gender = "female"
        elif "non-binary" in low or "nonbinary" in low:
            gender = "non-binary"
        elif "male" in low:  # checked after "female" (substring)
            gender = "male"
        else:
            gender = "female" if (h % 100) < int(self.female_share_unconstrained * 100) else "male"

        # Skin tone: honour an explicit request; otherwise default light.
        if "black" in low:
            tone_label, fitz = "dark", 6
        elif "white" in low:
            tone_label, fitz = "light", 2
        elif "south asian" in low:
            tone_label, fitz = "medium-dark", 5
        elif "asian" in low:
            tone_label, fitz = "medium", 4
        elif "latino" in low or "latina" in low:
            tone_label, fitz = "medium-dark", 5
        else:
            tone_label, fitz = "light", 2

        # Quality/setting signals, with a mild, illustrative stereotype pattern:
        # some female depictions land in a less professional setting.
        if gender == "female" and (h % 3 == 0):
            setting, equipment, quality, severity = "local_field", "amateur", 6.4, 0.4
            stereotypes = ["female player depicted in a less professional setting"]
        else:
            setting, equipment, quality, severity = "professional_stadium", "professional_kit", 8.2, 0.0
            stereotypes = []

        return {
            "perceived_gender": gender,
            "skin_tone_fitzpatrick": fitz,
            "skin_tone_label": tone_label,
            "setting": setting,
            "equipment_quality": equipment,
            "pose_type": "action_competitive",
            "age_presentation": "adult",
            "image_quality_score": quality,
            "stereotype_severity": severity,
            "stereotypes_detected": stereotypes,
            "description": f"[stub] {gender} soccer subject, {tone_label} skin tone, {setting}",
        }

    async def generate(
        self,
        prompt: str,
        config: ImageGenerationConfig | None = None,
    ) -> GeneratedImage:
        config = config or ImageGenerationConfig()
        h = self._hash(prompt)
        analysis = self._recorded_analysis(prompt)
        latency_ms = 40.0 + float(h % 160)

        # Refusal path: no bytes, a content-policy reason, no baked analysis.
        if self.refuse_nonbinary and analysis["perceived_gender"] == "non-binary":
            return GeneratedImage(
                prompt=prompt,
                model_info=self.get_model_info(),
                generation_config=config,
                image_data=None,
                revised_prompt=None,
                latency_ms=latency_ms,
                metadata={
                    "refused": True,
                    "refusal_reason": "content_policy: request could not be rendered",
                    "status": "REFUSED_CONTENT_POLICY",
                },
            )

        return GeneratedImage(
            prompt=prompt,
            model_info=self.get_model_info(),
            generation_config=config,
            image_data=_PNG_1X1,
            # A real hosted model may rewrite the prompt; record what was "used"
            # so the pipeline can measure rewriter effects.
            revised_prompt=f"{prompt} — rendered by stub",
            latency_ms=latency_ms,
            metadata={"status": "SUCCESS", "stub_analysis": analysis},
        )
