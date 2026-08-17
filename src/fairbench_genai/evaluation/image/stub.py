"""Offline stubs for the image evaluation layer.

StubVisionAnalyzer and StubCLIPEvaluator are drop-in, duck-typed replacements
for VisionAnalyzer (Claude Vision) and CLIPEvaluator. They require no API key
and no model download. Instead of calling a model, they replay the demographic
signal that StubImageAdapter baked into ``image.metadata["stub_analysis"]``.

Together with StubImageAdapter, they let ImageBenchEngine.evaluate() run the
full Chapter 5 pipeline end to end, deterministically and offline:

    engine.evaluate(
        model=StubImageAdapter(),
        scenarios=["soccer_player_action"],
        vision_analyzer=StubVisionAnalyzer(),
        clip_evaluator=StubCLIPEvaluator(),
    )
"""

from __future__ import annotations

import hashlib

from fairbench_genai.core.image_types import GeneratedImage, ImageAnalysis


class StubVisionAnalyzer:
    """Replays the baked-in demographic reading; no Anthropic API key needed.

    Interface-compatible with VisionAnalyzer: exposes ``model`` and the async
    ``analyze`` / ``analyze_batch`` methods that ImageBenchEngine calls.
    """

    def __init__(self, model: str = "stub-vision") -> None:
        self.model = model

    async def analyze(self, image: GeneratedImage) -> ImageAnalysis:
        if not image.has_image() or image.metadata.get("refused"):
            return ImageAnalysis(
                perceived_gender="unknown",
                description="[stub] refused or empty output; no analysis",
                analysis_confidence=0.0,
                analysis_source="stub",
            )
        sa = image.metadata.get("stub_analysis", {})
        return ImageAnalysis(
            perceived_gender=sa.get("perceived_gender", "unknown"),
            skin_tone_fitzpatrick=sa.get("skin_tone_fitzpatrick"),
            skin_tone_label=sa.get("skin_tone_label", "unknown"),
            setting=sa.get("setting", "unknown"),
            equipment_quality=sa.get("equipment_quality", "unknown"),
            pose_type=sa.get("pose_type", "unknown"),
            age_presentation=sa.get("age_presentation", "unknown"),
            image_quality_score=sa.get("image_quality_score"),
            stereotype_severity=sa.get("stereotype_severity", 0.0),
            stereotypes_detected=sa.get("stereotypes_detected", []),
            description=sa.get("description", ""),
            analysis_confidence=1.0,
            analysis_source="stub",
        )

    async def analyze_batch(self, images: list[GeneratedImage]) -> list[ImageAnalysis]:
        return [await self.analyze(img) for img in images]


class StubCLIPEvaluator:
    """Deterministic visual embeddings + probe similarities; no CLIP download.

    Interface-compatible with CLIPEvaluator: exposes ``model_name`` and the
    async ``analyze_batch`` returning ``(embedding, similarities)`` per image.
    The embedding depends on the recorded gender so that CDS (counterfactual
    distance) and ODE (diversity) have a real, if synthetic, signal to work on.
    """

    def __init__(self, model_name: str = "stub-clip") -> None:
        self.model_name = model_name

    def _embedding(self, gender: str, prompt: str) -> list[float]:
        base = {
            "male": [1.0, 0.0, 0.0],
            "female": [0.0, 1.0, 0.0],
            "non-binary": [0.0, 0.0, 1.0],
        }.get(gender, [0.3, 0.3, 0.3])
        # Small deterministic jitter so identical genders are not identical points.
        h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
        jitter = [((h >> (i * 8)) % 100) / 1000.0 for i in range(3)]
        return [b + j for b, j in zip(base, jitter)]

    async def analyze_batch(
        self, images: list[GeneratedImage]
    ) -> list[tuple[list[float] | None, dict[str, float]]]:
        results: list[tuple[list[float] | None, dict[str, float]]] = []
        for img in images:
            if not img.has_image() or img.metadata.get("refused"):
                results.append((None, {}))
                continue
            gender = img.metadata.get("stub_analysis", {}).get("perceived_gender", "unknown")
            emb = self._embedding(gender, img.prompt)
            sims = {
                "gender_male": 0.9 if gender == "male" else 0.1,
                "gender_female": 0.9 if gender == "female" else 0.1,
            }
            results.append((emb, sims))
        return results
