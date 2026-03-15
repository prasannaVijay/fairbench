"""Abstract storage backend interface."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from fairbench.core.types import EvaluationRun, RunStatus


class RunFilters:
    """Filters for querying evaluation runs."""

    def __init__(
        self,
        status: RunStatus | None = None,
        model_name: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        scenario_set: str | None = None,
    ) -> None:
        self.status = status
        self.model_name = model_name
        self.created_after = created_after
        self.created_before = created_before
        self.scenario_set = scenario_set


class RunSummary:
    """Summary of an evaluation run for listing."""

    def __init__(
        self,
        run_id: str,
        status: RunStatus,
        model_name: str,
        scenario_sets: list[str],
        created_at: datetime,
        completed_at: datetime | None = None,
        metric_summary: dict[str, float] | None = None,
    ) -> None:
        self.run_id = run_id
        self.status = status
        self.model_name = model_name
        self.scenario_sets = scenario_sets
        self.created_at = created_at
        self.completed_at = completed_at
        self.metric_summary = metric_summary or {}


class StorageBackend(ABC):
    """Abstract base class for storage backends.

    Storage backends persist evaluation runs, allowing:
    - Saving and loading complete runs
    - Querying runs by various criteria
    - Storing artifacts (reports, embeddings caches)
    """

    @abstractmethod
    async def save_run(self, run: EvaluationRun) -> str:
        """Save an evaluation run.

        Args:
            run: The evaluation run to save.

        Returns:
            The run ID.
        """
        pass

    @abstractmethod
    async def get_run(self, run_id: str) -> EvaluationRun | None:
        """Get an evaluation run by ID.

        Args:
            run_id: The run ID.

        Returns:
            The evaluation run, or None if not found.
        """
        pass

    @abstractmethod
    async def update_run(self, run: EvaluationRun) -> None:
        """Update an existing evaluation run.

        Args:
            run: The evaluation run with updated data.
        """
        pass

    @abstractmethod
    async def delete_run(self, run_id: str) -> bool:
        """Delete an evaluation run.

        Args:
            run_id: The run ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        pass

    @abstractmethod
    async def list_runs(
        self,
        filters: RunFilters | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RunSummary]:
        """List evaluation runs.

        Args:
            filters: Optional filters to apply.
            limit: Maximum number of runs to return.
            offset: Number of runs to skip.

        Returns:
            List of run summaries.
        """
        pass

    @abstractmethod
    async def save_artifact(
        self, run_id: str, name: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        """Save a binary artifact.

        Args:
            run_id: The associated run ID.
            name: Name for the artifact.
            data: The binary data.
            content_type: MIME type of the artifact.

        Returns:
            URI or path to the saved artifact.
        """
        pass

    @abstractmethod
    async def get_artifact(self, run_id: str, name: str) -> bytes | None:
        """Get a saved artifact.

        Args:
            run_id: The associated run ID.
            name: Name of the artifact.

        Returns:
            The artifact data, or None if not found.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the storage backend and release resources."""
        pass
