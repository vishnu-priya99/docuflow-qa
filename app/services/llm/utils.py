from __future__ import annotations

import json
import re


def strip_code_fence(text: str) -> str:
    """Remove a leading/trailing ``` fence some models wrap SQL/code in."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_index_list(raw: str, *, count: int) -> list[int]:
    """Parses a reranker's "[3, 0, 5]"-style response into a list of valid,
    de-duplicated indices into a 0..count-1 candidate list. Never raises -
    a malformed/empty response just yields an empty list, so callers can
    fall back to the original (pre-rerank) order."""
    text = strip_code_fence(raw)
    match = re.search(r"\[[^\]]*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []

    seen: set[int] = set()
    result: list[int] = []
    for item in parsed:
        if isinstance(item, bool) or not isinstance(item, int):
            continue
        if 0 <= item < count and item not in seen:
            seen.add(item)
            result.append(item)
    return result
