from __future__ import annotations

import re

from app.services.llm.base import LLMProvider

_TRAILING_NO_QUERY_RE = re.compile(r"\s*NO_QUERY\s*$")


async def generate_sql(llm: LLMProvider, *, question: str, schema_description: str) -> str:
    if not schema_description.strip():
        return "NO_QUERY"
    raw = await llm.generate_sql(question=question, schema_description=schema_description)
    return _clean_sql(raw)


def _clean_sql(raw: str) -> str:
    """A local model can hedge on a multi-part question by writing a
    genuinely valid SQL query and then appending the NO_QUERY refusal
    marker onto the end anyway - as if trying to both answer and refuse
    at once (observed live and reproducible: "SELECT COUNT(*) FROM ...
    WHERE ... NO_QUERY", which fails to parse as either). If NO_QUERY
    only trails other real content rather than being the model's entire,
    deliberate answer, strip it and keep the real query - sql_validator
    still independently checks whatever's left is actually valid SQL, so
    this doesn't weaken validation, it just stops a real generated answer
    from being thrown away over a formatting mistake."""
    text = raw.strip()
    if text == "NO_QUERY":
        return text
    return _TRAILING_NO_QUERY_RE.sub("", text).strip()
