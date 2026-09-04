"""Artifact storage for the Chapter 8 pilot.

A run leaves behind a directory, not a number. The directory holds the images,
the classifier labels, the metric results and the summary, so that anyone
reading a score months later can see the outputs it came from.

This is a local filesystem store, which is the right size for a pilot. Chapter 9
replaces it with a registry that can query across runs and track a metric's
history, and the interface here is the one that registry has to satisfy.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


def _jsonable(value: Any) -> Any:
    """Convert results, dataclasses and paths into something json can write."""
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "name") and hasattr(value, "value") and hasattr(value, "details"):
        return {
            "name": value.name,
            "value": float(value.value),
            "details": _jsonable(value.details),
        }
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass
class Run:
    """One benchmark run and the directory it owns."""

    run_id: str
    scenario_id: str
    model_id: str
    run_timestamp: datetime
    path: Path
    written: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "images").mkdir(exist_ok=True)
        self._write_json(
            "run.json",
            {
                "run_id": self.run_id,
                "scenario_id": self.scenario_id,
                "model_id": self.model_id,
                "run_timestamp": self.run_timestamp.isoformat(),
            },
        )

    def _write_json(self, name: str, payload: Any) -> Path:
        target = self.path / name
        target.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")
        self.written.append(name)
        return target

    def write_images(self, image_dir: str | Path) -> Path:
        """Move the generated images and their metadata into the run directory.

        If the images were generated straight into the run directory this is a
        no-op, which is the usual case; the copy exists for the run that
        generated into a scratch directory first.
        """
        source = Path(image_dir)
        target = self.path / "images"
        if source.resolve() == target.resolve():
            return target
        target.mkdir(parents=True, exist_ok=True)
        for item in sorted(source.iterdir()):
            if item.is_file():
                shutil.copy2(item, target / item.name)
        return target

    def write_classifier_results(self, classifier_results: Sequence[Any]) -> Path:
        return self._write_json("classifier_results.json", list(classifier_results))

    def write_metrics(self, metrics: dict[str, Any]) -> Path:
        return self._write_json("metrics.json", metrics)

    def write_summary(self, summary: str | dict[str, Any]) -> Path:
        """Write the human-readable metrics summary.

        The summary is the file a product owner opens. It carries the metric
        values, the band each one landed in, the overall recommendation, and the
        flags worth reading first.
        """
        target = self.path / "metrics_summary.yaml"
        text = summary if isinstance(summary, str) else _render_summary(summary)
        target.write_text(text, encoding="utf-8")
        self.written.append("metrics_summary.yaml")
        return target


def _render_summary(summary: dict[str, Any]) -> str:  # pragma: no cover - convenience path
    import yaml

    return yaml.safe_dump(_jsonable(summary), sort_keys=False)


class RunArtifactStore:
    """Creates and names run directories under a base path."""

    def __init__(self, base_path: str | Path = "runs/") -> None:
        self.base_path = Path(base_path)

    def create_run(
        self,
        scenario_id: str,
        model_id: str,
        run_timestamp: datetime,
    ) -> Run:
        stamp = run_timestamp.strftime("%Y%m%d_%H%M%S")
        return Run(
            run_id=f"{scenario_id}_{stamp}",
            scenario_id=scenario_id,
            model_id=model_id,
            run_timestamp=run_timestamp,
            path=self.base_path / stamp,
        )

    def runs(self) -> list[Path]:
        if not self.base_path.exists():
            return []
        return sorted(p for p in self.base_path.iterdir() if p.is_dir())


__all__ = ["Run", "RunArtifactStore"]
