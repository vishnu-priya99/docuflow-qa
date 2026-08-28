from __future__ import annotations

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, new_uuid
from app.db.types import GUID, JSONVariant


class Message(TimestampMixin, Base):
    """A single chat turn (user question or assistant answer)."""

    __tablename__ = "messages"

    message_id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    session_id: Mapped[str] = mapped_column(
        GUID, ForeignKey("sessions.session_id", ondelete="CASCADE"), index=True, nullable=False
    )
    user_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)

    role: Mapped[str] = mapped_column(String(16), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # SEMANTIC | STRUCTURED | HYBRID - set on assistant messages by the router.
    question_type: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # Source references shown with the answer, e.g.
    # [{"filename": "a.pdf", "page_start": 12, "section": "..."}]
    sources: Mapped[list | None] = mapped_column(JSONVariant, nullable=True)

    session: Mapped["ChatSession"] = relationship(back_populates="messages")  # noqa: F821
