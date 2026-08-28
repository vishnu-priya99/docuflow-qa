from __future__ import annotations

from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Storage abstraction for original uploaded files.

    Implementations must scope every path under session_id/file_id so
    deleting a session's directory removes every original file it owns.
    """

    @abstractmethod
    async def save(self, *, session_id: str, file_id: str, filename: str, content: bytes) -> str:
        """Persist ``content`` and return a backend-specific storage path."""

    @abstractmethod
    async def read(self, storage_path: str) -> bytes:
        """Return the raw bytes stored at ``storage_path``."""

    @abstractmethod
    async def delete_session(self, session_id: str) -> None:
        """Delete every file belonging to ``session_id``."""
