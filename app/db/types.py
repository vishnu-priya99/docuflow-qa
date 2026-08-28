"""Shared column type helpers."""
from __future__ import annotations

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import String, TypeDecorator

JSONVariant = JSONB


class GUID(TypeDecorator):
    """A plain string-based identifier column (application-generated ids,
    not Postgres' native UUID type - keeps id generation simple and
    consistent everywhere it's used)."""

    impl = String(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):  # noqa: ANN001
        return str(value) if value is not None else None

    def process_result_value(self, value, dialect):  # noqa: ANN001
        return value
