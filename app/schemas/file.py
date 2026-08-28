from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class FileOut(BaseModel):
    file_id: str
    filename: str
    file_type: str
    status: str
    size_bytes: int
    error_message: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class FileListOut(BaseModel):
    files: list[FileOut]
