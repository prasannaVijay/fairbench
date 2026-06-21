"""CLIPEvaluator: CLIP-based visual embeddings and text-image similarity probes.

Two roles:
  1. Visual embeddings — fed into CDS (cosine distance between counterfactual
     image pairs) and ODE (diversity across the embedding space).
  2. Text-image similarity probes — zero-shot demographic classification
     complementing the Claude Vision analysis. Used to cross-validate and compare.

The CLIP ViT-B/32 model (~150MB) is loaded lazily on first use.
"""

from typing import Any

import numpy as np

from fairbench.core.exceptions import EvaluationError
from fairbench.core.image_types import GeneratedImage

# Demographic probe pairs for soccer — can be overridden per use case
SOCCER_PROBES = {
    # Gender probes
    "gender_male": "a photo of a male soccer player",
    "gender_female": "a photo of a female soccer player",
    # Professional setting probes
    "setting_professional": "a professional soccer player in a stadium",
    "setting_amateur": "an amateur soccer player in a local park",
    # Action quality probes
    "pose_competitive": "a soccer player in competitive action during a match",
    "pose_casual": "a person casually kicking a soccer ball",
    # Equipment quality probes
    "equipment_pro": "a soccer player wearing a professional team kit",
    "equipment_amateur": "a person in generic sports clothes playing soccer",
    # Age probes
    "age_youth": "a young child playing soccer",
    "age_adult": "an adult soccer player",
}


class CLIPEvaluator:
    """Computes CLIP visual embeddings and text-image similarity probes.

    Args:
        model_name: CLIP model variant. Default is ViT-B/32 (fast, ~150MB).
                    Use "ViT-L/14" for higher quality (slower, ~900MB).
        device: Compute device ("cpu", "cuda", "mps").
        probes: Dict of probe_name → probe_text. Defaults to SOCCER_PROBES.
        fallback_on_error: Return zeros on model load / inference error.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cpu",
        probes: dict[str, str] | None = None,
        fallback_on_error: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.probes = probes if probes is not None else SOCCER_PROBES
        self.fallback_on_error = fallback_on_error
        self._model: Any = None
        self._preprocess: Any = None
        self._text_embeddings: dict[str, Any] = {}

    def _load_model(self) -> tuple[Any, Any]:
        """Lazily load CLIP model and preprocessing pipeline."""
        if self._model is None:
            try:
                import clip  # openai-clip package
                import torch
            except ImportError:
                raise EvaluationError(
                    "openai-clip not installed. Run: pip install openai-clip"
                )
            self._model, self._preprocess = clip.load(self.model_name, device=self.device)
        return self._model, self._preprocess

    def _get_text_embeddings(self) -> dict[str, Any]:
        """Pre-compute and cache text embeddings for all probes."""
        if not self._text_embeddings and self.probes:
            try:
                import clip
                import torch

                model, _ = self._load_model()
                texts = list(self.probes.values())
                tokens = clip.tokenize(texts).to(self.device)

                with torch.no_grad():
                    text_feats = model.encode_text(tokens)
                    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

                for key, feat in zip(self.probes.keys(), text_feats):
                    self._text_embeddings[key] = feat

            except Exception as e:
                if not self.fallback_on_error:
                    raise EvaluationError(f"CLIP text embedding failed: {e}") from e

        return self._text_embeddings

    async def embed(self, image: GeneratedImage) -> list[float] | None:
        """Compute a CLIP visual embedding for an image.

        Returns a normalized float list suitable for cosine similarity.
        Returns None if the image has no content or CLIP fails.
        """
        import asyncio

        def _sync() -> list[float] | None:
            return self._embed_sync(image)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync)

    def _embed_sync(self, image: GeneratedImage) -> list[float] | None:
        """Synchronous CLIP embedding computation."""
        try:
            import torch
            from PIL import Image as PILImage

            model, preprocess = self._load_model()
            pil_img = self._load_pil(image)
            if pil_img is None:
                return None

            img_tensor = preprocess(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = model.encode_image(img_tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)

            return feat.squeeze().cpu().numpy().tolist()

        except Exception as e:
            if not self.fallback_on_error:
                raise EvaluationError(f"CLIP embedding failed: {e}") from e
            return None

    async def similarity_probes(
        self, image: GeneratedImage
    ) -> dict[str, float]:
        """Compute cosine similarity between the image and all text probes.

        Returns a dict of probe_name → similarity score (0–1).
        Higher means the image more closely matches that probe description.
        """
        import asyncio

        def _sync() -> dict[str, float]:
            return self._similarity_probes_sync(image)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync)

    def _similarity_probes_sync(self, image: GeneratedImage) -> dict[str, float]:
        """Synchronous similarity probe computation."""
        try:
            import torch

            model, preprocess = self._load_model()
            pil_img = self._load_pil(image)
            if pil_img is None:
                return {}

            img_tensor = preprocess(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                img_feat = model.encode_image(img_tensor)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            text_embeds = self._get_text_embeddings()
            if not text_embeds:
                return {}

            similarities = {}
            for probe_name, text_feat in text_embeds.items():
                sim = (img_feat @ text_feat.unsqueeze(-1)).squeeze().item()
                similarities[probe_name] = float(np.clip(sim, -1.0, 1.0))

            return similarities

        except Exception as e:
            if not self.fallback_on_error:
                raise EvaluationError(f"CLIP similarity probes failed: {e}") from e
            return {}

    def _load_pil(self, image: GeneratedImage) -> Any:
        """Load a PIL Image from any available image source."""
        try:
            from PIL import Image as PILImage
            import io

            if image.image_data:
                return PILImage.open(io.BytesIO(image.image_data)).convert("RGB")
            elif image.image_path and image.image_path.exists():
                return PILImage.open(image.image_path).convert("RGB")
            elif image.image_url:
                # Synchronous HTTP fetch for PIL loading (called from executor)
                import ssl
                import urllib.request
                ctx = ssl.create_default_context()
                with urllib.request.urlopen(image.image_url, timeout=30, context=ctx) as resp:
                    return PILImage.open(io.BytesIO(resp.read())).convert("RGB")
        except Exception:
            pass
        return None

    async def analyze_batch(
        self, images: list[GeneratedImage]
    ) -> list[tuple[list[float] | None, dict[str, float]]]:
        """Analyze a batch of images and return (embedding, probe_similarities) tuples."""
        import asyncio

        async def _one(img: GeneratedImage) -> tuple[list[float] | None, dict[str, float]]:
            emb = await self.embed(img)
            probes = await self.similarity_probes(img)
            return emb, probes

        return list(await asyncio.gather(*[_one(img) for img in images]))

    def infer_gender_from_probes(self, similarities: dict[str, float]) -> str:
        """Infer perceived gender label from CLIP probe similarities.

        Returns "male", "female", or "ambiguous".
        """
        male_score = similarities.get("gender_male", 0.0)
        female_score = similarities.get("gender_female", 0.0)
        diff = abs(male_score - female_score)

        if diff < 0.02:
            return "ambiguous"
        return "male" if male_score > female_score else "female"

    def infer_setting_quality(self, similarities: dict[str, float]) -> str:
        """Infer setting quality from CLIP probe similarities."""
        pro_score = similarities.get("setting_professional", 0.0)
        amateur_score = similarities.get("setting_amateur", 0.0)
        if pro_score > amateur_score + 0.02:
            return "professional_stadium"
        elif amateur_score > pro_score + 0.02:
            return "local_field"
        return "generic"
