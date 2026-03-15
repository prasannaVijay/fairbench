"""Storage backends for FAIRBench."""

from fairbench.storage.base import StorageBackend
from fairbench.storage.sqlite import SQLiteBackend

__all__ = [
    "StorageBackend",
    "SQLiteBackend",
]
