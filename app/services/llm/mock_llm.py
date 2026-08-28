from __future__ import annotations

import re

from app.services.llm.base import LLMProvider

_STRUCTURED_KEYWORDS = (
    "count",
    "sum",
    "average",
    "avg",
    "total",
    "how many",
    "unique",
    "maximum",
    "minimum",
    "mean",
    "group by",
)
_SEMANTIC_KEYWORDS = (
    "summary",
    "summarize",
    "recommend",
    "explain",
    "describe",
    "what does",
    "main point",
    "discuss",
    "says about",
    "according to",
)
_DISCOVERY_KEYWORDS = ("find the", "which file", "which sheet", "find and")


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


class MockLLMProvider(LLMProvider):
    """Deterministic, offline stand-in for a real LLM, used in tests.

    Prefers exact canned responses (keyed by normalized question text) when
    provided, and otherwise falls back to small, honest heuristics. This is
    NOT a general NL2SQL engine - test cases that exercise SQL generation
    should inject the expected SQL explicitly via ``sql_answers``.
    """

    def __init__(
        self,
        *,
        route_answers: dict[str, str] | None = None,
        sql_answers: dict[str, str] | None = None,
        answer_answers: dict[str, str] | None = None,
        rerank_answers: dict[str, list[int]] | None = None,
    ) -> None:
        self._route_answers = {_norm(k): v for k, v in (route_answers or {}).items()}
        self._sql_answers = {_norm(k): v for k, v in (sql_answers or {}).items()}
        self._answer_answers = {_norm(k): v for k, v in (answer_answers or {}).items()}
        self._rerank_answers = {_norm(k): v for k, v in (rerank_answers or {}).items()}

    async def classify_question(
        self, *, question: str, has_documents: bool, has_structured_data: bool
    ) -> str:
        key = _norm(question)
        if key in self._route_answers:
            return self._route_answers[key]

        q = key
        has_structured_kw = any(kw in q for kw in _STRUCTURED_KEYWORDS)
        has_semantic_kw = any(kw in q for kw in _SEMANTIC_KEYWORDS)
        has_discovery_kw = any(kw in q for kw in _DISCOVERY_KEYWORDS)

        if has_structured_kw and has_structured_data and (has_semantic_kw or has_discovery_kw) and has_documents:
            return "HYBRID"
        if has_structured_kw and has_structured_data:
            return "STRUCTURED"
        if has_documents:
            return "SEMANTIC"
        if has_structured_data:
            return "STRUCTURED"
        return "SEMANTIC"

    async def generate_sql(self, *, question: str, schema_description: str) -> str:
        key = _norm(question)
        if key in self._sql_answers:
            return self._sql_answers[key]
        return _heuristic_sql(question=question, schema_description=schema_description)

    async def generate_answer(self, *, question: str, context: str, question_type: str) -> str:
        key = _norm(question)
        if key in self._answer_answers:
            return self._answer_answers[key]
        if not context.strip():
            return "I couldn't find that information in the uploaded files."
        # Best-effort default: surface a DB result verbatim if present,
        # otherwise return a short excerpt of the retrieved evidence.
        db_match = re.search(r"Database result:\s*(.+)", context)
        if db_match:
            return db_match.group(1).strip().splitlines()[0]
        first_block = context.strip().split("\n\n")[0]
        lines = [ln for ln in first_block.splitlines() if not ln.startswith("[Source:")]
        text = " ".join(lines).strip()
        return text[:300] if text else "I couldn't find that information in the uploaded files."

    async def rerank(self, *, question: str, candidates: list[str], top_n: int) -> list[int]:
        key = _norm(question)
        if key in self._rerank_answers:
            return self._rerank_answers[key][:top_n]
        if not candidates:
            return []
        # Deterministic lexical-overlap heuristic (word overlap with the
        # question) - not a real semantic reranker, but reproducible and
        # dependency-free for offline tests. Real providers use an LLM call.
        q_words = set(_norm(question).split())
        scored = [
            (i, len(q_words & set(_norm(text).split())))
            for i, text in enumerate(candidates)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [i for i, score in scored if score > 0][:top_n]


def _heuristic_sql(*, question: str, schema_description: str) -> str:
    q = _norm(question).rstrip("?.! ")
    # Anchored to line start so a table/sheet name mentioned in the leading
    # "-- Sheet: ..." comment line isn't mistaken for the real table name.
    table_match = re.search(r"^(\w+)\(", schema_description, re.MULTILINE)
    table = table_match.group(1) if table_match else "data"
    columns = re.findall(r'"([^"]+)"\s+\w+', schema_description)

    def find_column(fragment: str) -> str | None:
        frag = fragment.lower()
        for col in columns:
            if col.lower() in frag or frag in col.lower():
                return col
        return None

    m = re.search(r"unique\s+(\w+)|distinct\s+(\w+)", q)
    if m:
        target = m.group(1) or m.group(2)
        col = find_column(target) or (columns[0] if columns else target)
        return f'SELECT COUNT(DISTINCT "{col}") FROM {table};'

    original = question.strip().rstrip("?.! ")

    m = re.search(r"(average|avg|mean)\s+(\w+)", q)
    if m:
        col = find_column(m.group(2)) or (columns[0] if columns else m.group(2))
        where = _extract_where(original, columns)
        return f'SELECT AVG("{col}") FROM {table}{where};'

    m = re.search(r"(total|sum)\s+(?:of\s+)?(\w+)", q)
    if m:
        col = find_column(m.group(2)) or (columns[0] if columns else m.group(2))
        where = _extract_where(original, columns)
        return f'SELECT SUM("{col}") FROM {table}{where};'

    return "NO_QUERY"


def _extract_where(original_question: str, columns: list[str]) -> str:
    """Looks for an explicit/implicit filter in the ORIGINAL (case-preserved)
    question text, since column values are matched case-sensitively."""
    m = re.search(r"where\s+(\w+)\s*=\s*'?([\w\s]+?)'?$", original_question, re.IGNORECASE)
    if m:
        col = m.group(1)
        for c in columns:
            if c.lower() == col.lower():
                return f" WHERE \"{c}\" = '{m.group(2).strip()}'"
    m = re.search(r"\bin\s+(?:the\s+)?([A-Za-z]+)\b", original_question, re.IGNORECASE)
    if m:
        value = m.group(1)
        for c in columns:
            if c.lower() not in ("name", "id"):
                return f" WHERE \"{c}\" = '{value}'"
    return ""
