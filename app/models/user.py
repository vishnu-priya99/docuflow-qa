from __future__ import annotations

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class User(TimestampMixin, Base):
    """A user, identified solely by the id they type on the login page.

    No password/auth is implemented per the product spec - user_id is the
    login. Every downstream record traces back to this id.
    """

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)

    sessions: Mapped[list["ChatSession"]] = relationship(  # noqa: F821
        back_populates="user", cascade="all, delete-orphan"
    )
