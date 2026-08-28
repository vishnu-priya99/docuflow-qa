from __future__ import annotations

import shutil
from pathlib import Path

from app.core.logging import get_logger
from app.services.storage.base import StorageBackend

logger = get_logger(__name__)


class LocalStorageBackend(StorageBackend):
    """Stores original files on local disk under ``<root>/<session_id>/<file_id>_<filename>``."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        return self.root / session_id

    async def save(self, *, session_id: str, file_id: str, filename: str, content: bytes) -> str:
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        safe_name = filename.replace("/", "_").replace("\\", "_")
        target = session_dir / f"{file_id}_{safe_name}"
        target.write_bytes(content)
        return str(target)

    async def read(self, storage_path: str) -> bytes:
        return Path(storage_path).read_bytes()

    async def delete_session(self, session_id: str) -> None:
        session_dir = self._session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)
            logger.info("Deleted local storage directory for session %s", session_id)
