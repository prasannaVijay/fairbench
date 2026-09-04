"""The model access layer: prompts in, images and metadata out.

Two things are worth building correctly from the first run, and both are here.
Retry logic, because transient rate limits are expected rather than
exceptional. And metadata attachment, because a score that cannot be traced
back to the prompt behind it is a number nobody can act on.

The API key is read from the environment, never from the configuration file.
That keeps credentials out of version control and out of the run artifact, and
it is the difference between a benchmark you can share and one you cannot.

When no key is present the client replays a recorded run instead of calling the
provider. The replay walks the same code path - same variants, same metadata,
same files on disk - so that reading along costs nothing.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# A 1x1 transparent PNG. The recorded path writes real image files so that the
# classifier stage globs and reads exactly what it would in a live run; the
# pixels themselves carry nothing, since the recorded classifier scores stand in
# for the model's actual output.
_PLACEHOLDER_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000a49444154789c6360000002000100ffff03000006"
    "000557bfabd40000000049454e44ae426082"
)


class ModelAccessError(RuntimeError):
    """Raised when the provider cannot be reached or refuses a request."""


@dataclass
class GenerationResult:
    """One generated image, its metadata, and what it cost."""

    prompt: str
    image_bytes: bytes
    metadata: dict[str, Any]
    cost_usd: float
    elapsed_s: float

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        """Write the image and the metadata sidecar that travels with it."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.image_bytes)
        payload = dict(self.metadata)
        if metadata:
            payload.update(metadata)
        p.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return p


class ImageGenerationClient:
    """Thin wrapper over an image generation endpoint.

    ``api_key`` is passed in by the caller, which reads it from the environment.
    Passing ``None`` puts the client in replay mode against ``recorded_run``.
    """

    def __init__(
        self,
        provider: str,
        model_id: str,
        api_key: str | None = None,
        endpoint: str | None = None,
        parameters: dict[str, Any] | None = None,
        cost_per_image_usd: float = 0.0,
        recorded_run: str | Path | None = None,
    ) -> None:
        self.provider = provider
        self.model_id = model_id
        self.api_key = api_key
        self.endpoint = endpoint
        self.parameters = parameters or {}
        self.cost_per_image_usd = cost_per_image_usd
        self.recorded: dict[str, dict[str, Any]] = {}
        self.recorded_meta: dict[str, Any] = {}
        if recorded_run is not None:
            self._load_recorded(Path(recorded_run))

    @property
    def is_replaying(self) -> bool:
        return self.api_key is None

    def _load_recorded(self, path: Path) -> None:
        if not path.exists():
            raise ModelAccessError(f"recorded run not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.recorded_meta = payload.get("run", {})
        self.recorded = {rec["image_id"]: rec for rec in payload["records"]}

    def generate(
        self,
        prompt: str,
        n: int = 1,
        retry_on_rate_limit: bool = True,
        max_retries: int = 3,
        image_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GenerationResult:
        """Generate one image, retrying through transient rate limits."""
        if self.is_replaying:
            return self._replay(prompt, image_id, metadata)
        return self._call_provider(prompt, n, retry_on_rate_limit, max_retries, metadata)

    def _replay(
        self, prompt: str, image_id: str | None, metadata: dict[str, Any] | None
    ) -> GenerationResult:
        if image_id is None or image_id not in self.recorded:
            raise ModelAccessError(f"no recorded output for image_id {image_id!r}")
        record = self.recorded[image_id]
        payload = dict(record.get("metadata", {}))
        if metadata:
            payload.update(metadata)
        payload["image_id"] = record["image_id"]
        payload["classifier_scores"] = record["classifier_scores"]
        payload["harm_scores"] = record.get("harm_scores", {})
        payload["service"] = record.get("service", {})
        return GenerationResult(
            prompt=prompt,
            image_bytes=_PLACEHOLDER_PNG,
            metadata=payload,
            cost_usd=self.cost_per_image_usd,
            elapsed_s=float(record.get("elapsed_s", 0.0)),
        )

    def _call_provider(
        self,
        prompt: str,
        n: int,
        retry_on_rate_limit: bool,
        max_retries: int,
        metadata: dict[str, Any] | None,
    ) -> GenerationResult:
        # Kept deliberately small. A live run needs httpx and a provider account;
        # everything the chapter walks through runs on the recorded path.
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - live path only
            raise ModelAccessError("httpx is required for live generation") from exc

        body = {"model": self.model_id, "prompt": prompt, "n": n, **self.parameters}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        attempt = 0
        started = time.monotonic()
        while True:  # pragma: no cover - live path only
            try:
                response = httpx.post(
                    self.endpoint, json=body, headers=headers, timeout=120.0
                )
                if response.status_code == 429 and retry_on_rate_limit and attempt < max_retries:
                    attempt += 1
                    time.sleep(2**attempt)
                    continue
                response.raise_for_status()
            except Exception as exc:
                if attempt >= max_retries:
                    raise ModelAccessError(
                        f"generation failed after {attempt} retries: {exc}"
                    ) from exc
                attempt += 1
                time.sleep(2**attempt)
                continue
            data = response.json()["data"][0]
            image_bytes = httpx.get(data["url"], timeout=120.0).content
            return GenerationResult(
                prompt=prompt,
                image_bytes=image_bytes,
                metadata=dict(metadata or {}),
                cost_usd=self.cost_per_image_usd,
                elapsed_s=time.monotonic() - started,
            )


def generate_images(
    prompt_variants: Sequence[Any],
    model_config: dict[str, Any],
    output_dir: str | Path,
    client: ImageGenerationClient | None = None,
) -> list[Path]:
    """Run every prompt variant through the model and save the outputs.

    Metadata travels with the image so that a metric can be traced back to the
    prompt that produced it.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if client is None:
        client = ImageGenerationClient(
            provider=model_config["provider"],
            model_id=model_config["model_id"],
            # The key lives in the environment or a secrets manager, never in
            # the config file.
            api_key=os.environ.get(model_config.get("api_key_env", "MODEL_API_KEY")),
            endpoint=model_config.get("endpoint"),
            parameters=model_config.get("parameters"),
            cost_per_image_usd=float(model_config.get("cost_per_image_usd", 0.0)),
            recorded_run=(model_config.get("fixtures") or {}).get("recorded_run"),
        )
    retry_cfg = model_config.get("retry") or {}
    written: list[Path] = []
    for variant in prompt_variants:
        for replicate_idx in range(variant.replicates):
            image_id = f"{variant.id}_rep_{replicate_idx:02d}"
            result = client.generate(
                prompt=variant.text,
                n=1,
                retry_on_rate_limit=bool(retry_cfg.get("retry_on_rate_limit", True)),
                max_retries=int(retry_cfg.get("max_retries", 3)),
                image_id=image_id,
                metadata={**variant.metadata, "replicate": replicate_idx},
            )
            written.append(result.save(path=output_dir / f"{image_id}.png"))
    return written


__all__ = [
    "GenerationResult",
    "ImageGenerationClient",
    "ModelAccessError",
    "generate_images",
]
