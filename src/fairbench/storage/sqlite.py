"""SQLite storage backend."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from fairbench.core.exceptions import StorageError
from fairbench.core.types import EvaluationRun, RunStatus
from fairbench.storage.base import RunFilters, RunSummary, StorageBackend


class SQLiteBackend(StorageBackend):
    """SQLite-based storage backend.

    Stores evaluation runs in a SQLite database with JSON serialization
    for complex fields. Artifacts are stored in a companion directory.
    """

    def __init__(
        self,
        db_path: str | Path = "~/.fairbench/fairbench.db",
    ) -> None:
        """Initialize the SQLite backend.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.db_path.parent / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)
        self._db: aiosqlite.Connection | None = None
        self._initialized = False

    async def _get_db(self) -> aiosqlite.Connection:
        """Get or create the database connection."""
        if self._db is None:
            self._db = await aiosqlite.connect(str(self.db_path))
            self._db.row_factory = aiosqlite.Row

        if not self._initialized:
            await self._init_schema()
            self._initialized = True

        return self._db

    async def _init_schema(self) -> None:
        """Initialize the database schema."""
        db = self._db
        if db is None:
            return

        await db.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_provider TEXT,
                scenario_sets TEXT NOT NULL,
                metrics_requested TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT,
                config_snapshot TEXT,
                outputs TEXT,
                metric_results TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
            CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model_name);
            CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);

            CREATE TABLE IF NOT EXISTS artifacts (
                run_id TEXT NOT NULL,
                name TEXT NOT NULL,
                content_type TEXT,
                path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (run_id, name),
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );
        """)
        await db.commit()

    async def save_run(self, run: EvaluationRun) -> str:
        """Save an evaluation run."""
        db = await self._get_db()

        try:
            # Serialize complex fields to JSON
            outputs_json = json.dumps(
                [o.model_dump(mode="json") for o in run.outputs]
            )
            metrics_json = json.dumps(
                [m.model_dump(mode="json") for m in run.metric_results]
            )
            config_json = json.dumps(run.config_snapshot)

            await db.execute(
                """
                INSERT OR REPLACE INTO runs (
                    id, status, model_name, model_provider, scenario_sets,
                    metrics_requested, created_at, started_at, completed_at,
                    error_message, config_snapshot, outputs, metric_results
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run.id),
                    run.status.value,
                    run.model_info.name,
                    run.model_info.provider,
                    json.dumps(run.scenario_sets),
                    json.dumps(run.metrics_requested),
                    run.created_at.isoformat(),
                    run.started_at.isoformat() if run.started_at else None,
                    run.completed_at.isoformat() if run.completed_at else None,
                    run.error_message,
                    config_json,
                    outputs_json,
                    metrics_json,
                ),
            )
            await db.commit()
            return str(run.id)
        except Exception as e:
            raise StorageError(f"Failed to save run: {e}") from e

    async def get_run(self, run_id: str) -> EvaluationRun | None:
        """Get an evaluation run by ID."""
        db = await self._get_db()

        try:
            cursor = await db.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            )
            row = await cursor.fetchone()

            if row is None:
                return None

            return self._row_to_run(row)
        except Exception as e:
            raise StorageError(f"Failed to get run: {e}") from e

    async def update_run(self, run: EvaluationRun) -> None:
        """Update an existing evaluation run."""
        # save_run uses INSERT OR REPLACE, so it handles updates
        await self.save_run(run)

    async def delete_run(self, run_id: str) -> bool:
        """Delete an evaluation run."""
        db = await self._get_db()

        try:
            # Delete artifacts first
            cursor = await db.execute(
                "SELECT path FROM artifacts WHERE run_id = ?", (run_id,)
            )
            rows = await cursor.fetchall()
            for row in rows:
                artifact_path = Path(row["path"])
                if artifact_path.exists():
                    artifact_path.unlink()

            await db.execute("DELETE FROM artifacts WHERE run_id = ?", (run_id,))
            cursor = await db.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            await db.commit()

            return cursor.rowcount > 0
        except Exception as e:
            raise StorageError(f"Failed to delete run: {e}") from e

    async def list_runs(
        self,
        filters: RunFilters | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RunSummary]:
        """List evaluation runs."""
        db = await self._get_db()

        # Build query
        query = "SELECT id, status, model_name, scenario_sets, created_at, completed_at, metric_results FROM runs"
        conditions = []
        params: list[Any] = []

        if filters:
            if filters.status:
                conditions.append("status = ?")
                params.append(filters.status.value)
            if filters.model_name:
                conditions.append("model_name = ?")
                params.append(filters.model_name)
            if filters.created_after:
                conditions.append("created_at >= ?")
                params.append(filters.created_after.isoformat())
            if filters.created_before:
                conditions.append("created_at <= ?")
                params.append(filters.created_before.isoformat())

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        try:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

            summaries = []
            for row in rows:
                # Parse metric results for summary
                metric_summary = {}
                if row["metric_results"]:
                    metrics = json.loads(row["metric_results"])
                    for m in metrics:
                        metric_summary[m["metric_name"]] = m["value"]

                summaries.append(
                    RunSummary(
                        run_id=row["id"],
                        status=RunStatus(row["status"]),
                        model_name=row["model_name"],
                        scenario_sets=json.loads(row["scenario_sets"]),
                        created_at=datetime.fromisoformat(row["created_at"]),
                        completed_at=(
                            datetime.fromisoformat(row["completed_at"])
                            if row["completed_at"]
                            else None
                        ),
                        metric_summary=metric_summary,
                    )
                )

            return summaries
        except Exception as e:
            raise StorageError(f"Failed to list runs: {e}") from e

    async def save_artifact(
        self, run_id: str, name: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        """Save a binary artifact."""
        db = await self._get_db()

        # Create artifact directory for this run
        run_artifacts_dir = self.artifacts_dir / run_id
        run_artifacts_dir.mkdir(exist_ok=True)

        # Sanitize name to prevent path traversal
        safe_name = Path(name).name  # strips any directory components
        if not safe_name or safe_name in (".", ".."):
            raise StorageError(f"Invalid artifact name: {name!r}")

        # Save file
        artifact_path = run_artifacts_dir / safe_name
        artifact_path.write_bytes(data)

        try:
            await db.execute(
                """
                INSERT OR REPLACE INTO artifacts (run_id, name, content_type, path, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, name, content_type, str(artifact_path), datetime.now(timezone.utc).isoformat()),
            )
            await db.commit()
            return str(artifact_path)
        except Exception as e:
            raise StorageError(f"Failed to save artifact: {e}") from e

    async def get_artifact(self, run_id: str, name: str) -> bytes | None:
        """Get a saved artifact."""
        db = await self._get_db()

        try:
            cursor = await db.execute(
                "SELECT path FROM artifacts WHERE run_id = ? AND name = ?",
                (run_id, name),
            )
            row = await cursor.fetchone()

            if row is None:
                return None

            artifact_path = Path(row["path"])
            if not artifact_path.exists():
                return None

            return artifact_path.read_bytes()
        except Exception as e:
            raise StorageError(f"Failed to get artifact: {e}") from e

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    def _row_to_run(self, row: aiosqlite.Row) -> EvaluationRun:
        """Convert a database row to an EvaluationRun."""
        from fairbench.core.types import (
            EvaluatedOutput,
            MetricResult,
            ModelInfo,
        )

        # Parse JSON fields
        outputs_data = json.loads(row["outputs"]) if row["outputs"] else []
        metrics_data = json.loads(row["metric_results"]) if row["metric_results"] else []
        config_data = json.loads(row["config_snapshot"]) if row["config_snapshot"] else {}

        # Reconstruct outputs
        outputs = [EvaluatedOutput.model_validate(o) for o in outputs_data]

        # Reconstruct metric results
        metric_results = [MetricResult.model_validate(m) for m in metrics_data]

        return EvaluationRun(
            id=row["id"],
            status=RunStatus(row["status"]),
            model_info=ModelInfo(
                name=row["model_name"],
                provider=row["model_provider"] or "unknown",
            ),
            scenario_sets=json.loads(row["scenario_sets"]),
            metrics_requested=json.loads(row["metrics_requested"]),
            outputs=outputs,
            metric_results=metric_results,
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=(
                datetime.fromisoformat(row["started_at"])
                if row["started_at"]
                else None
            ),
            completed_at=(
                datetime.fromisoformat(row["completed_at"])
                if row["completed_at"]
                else None
            ),
            config_snapshot=config_data,
            error_message=row["error_message"],
        )
