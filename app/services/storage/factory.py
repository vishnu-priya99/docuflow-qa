from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.services.storage.base import StorageBackend
from app.services.storage.local_storage import LocalStorageBackend


@lru_cache
def get_storage_backend() -> StorageBackend:
    settings = get_settings()
    if settings.storage_backend == "local":
        return LocalStorageBackend(settings.storage_local_path)
    raise ValueError(f"Unsupported storage backend: {settings.storage_backend}")
