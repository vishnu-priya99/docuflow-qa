from __future__ import annotations

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, new_uuid
from app.db.types import GUID

SUPPORTED_DOCUMENT_TYPES = {"pdf", "docx", "pptx", "txt"}
SUPPORTED_STRUCTURED_TYPES = {"xlsx", "csv"}
SUPPORTED_FILE_TYPES = SUPPORTED_DOCUMENT_TYPES | SUPPORTED_STRUCTURED_TYPES


class FileRecord(TimestampMixin, Base):
    """Metadata for one uploaded file. Original bytes live in object storage."""

    __tablename__ = "files"

    file_id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    session_id: Mapped[str] = mapped_column(
        GUID, ForeignKey("sessions.session_id", ondelete="CASCADE"), index=True, nullable=False
    )
    user_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)

    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)

    # pending -> processing -> ready | failed
    status: Mapped[str] = mapped_column(String(16), default="pending")
    error_message: Mapped[str | None] = mapped_column(String(2000), nullable=True)

    session: Mapped["ChatSession"] = relationship(back_populates="files")  # noqa: F821
