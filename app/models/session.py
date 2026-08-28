from __future__ import annotations

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, new_uuid
from app.db.types import GUID


class ChatSession(TimestampMixin, Base):
    """A chat/workspace session. All data belongs to exactly one session."""

    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    user_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("users.user_id", ondelete="CASCADE"), index=True, nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), default="New chat")

    user: Mapped["User"] = relationship(back_populates="sessions")  # noqa: F821
    messages: Mapped[list["Message"]] = relationship(  # noqa: F821
        back_populates="session", cascade="all, delete-orphan"
    )
    files: Mapped[list["FileRecord"]] = relationship(  # noqa: F821
        back_populates="session", cascade="all, delete-orphan"
    )
