"""Identifier sanitization for dynamically-created structured-data tables.

Every table/column name that ends up in raw SQL (DDL or generated-SQL
execution) is derived exclusively through these helpers, so it is always
restricted to [a-z0-9_] and cannot be used to break out of an identifier
position even though it's string-built rather than parameter-bound (DDL
identifiers can't be parameterized in SQL).
"""
from __future__ import annotations

import hashlib
import re

_INVALID = re.compile(r"[^a-z0-9_]")
_LEADING_DIGIT = re.compile(r"^[0-9]")

_RESERVED = {"select", "from", "where", "table", "order", "group", "by", "insert", "update", "delete"}


def sanitize_column_name(name: str, *, existing: set[str]) -> str:
    slug = _INVALID.sub("_", name.strip().lower().replace(" ", "_"))
    slug = re.sub(r"_+", "_", slug).strip("_") or "col"
    if _LEADING_DIGIT.match(slug):
        slug = f"col_{slug}"
    if slug in _RESERVED:
        slug = f"{slug}_col"
    base = slug
    i = 1
    while slug in existing:
        i += 1
        slug = f"{base}_{i}"
    existing.add(slug)
    return slug


def build_table_name(*, session_id: str, sheet_name: str, sheet_id: str) -> str:
    slug = _INVALID.sub("_", sheet_name.strip().lower().replace(" ", "_"))
    slug = re.sub(r"_+", "_", slug).strip("_")[:32] or "sheet"
    short_hash = hashlib.blake2b(f"{session_id}:{sheet_id}".encode(), digest_size=5).hexdigest()
    name = f"xlsx_{slug}_{short_hash}"
    return name[:63]
