"""Qdrant access layer.

Every point stored here carries ``user_id`` / ``session_id`` in its payload
and every read/write is filtered by them, so data from different users or
sessions can never mix (see search_* and delete_session below).

QDRANT_URL must be a real server (see .qdrant-portable/qdrant.exe for a
no-Docker local install). Tests use a separate QDRANT_COLLECTION so they
never touch production data on that same server.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def describe_point_location(payload: dict[str, Any]) -> str:
    """One-line, human-readable summary of a point's location/identity -
    used purely for the [RETRIEVAL]/[RERANK] demo/debug logs, not shown to
    the user. Location only - the full text is logged separately so it's
    never silently truncated."""
    if payload.get("content_type") == "excel_sheet":
        return f"sheet '{payload.get('sheet_name')}' ({payload.get('filename')})"

    loc_bits = []
    if payload.get("page_start") is not None:
        loc_bits.append(f"p.{payload['page_start']}")
    if payload.get("slide_number") is not None:
        loc_bits.append(f"slide {payload['slide_number']}")
    if payload.get("line_start") is not None:
        loc_bits.append(f"line {payload['line_start']}")
    if payload.get("section"):
        loc_bits.append(f"sec='{payload['section']}'")
    loc = " ".join(loc_bits)

    filename = payload.get("filename", "?")
    return f"{filename} {loc}" if loc else filename


@dataclass
class ChunkPayload:
    """Minimum + location metadata for one document chunk (spec section 3)."""

    user_id: str
    session_id: str
    file_id: str
    filename: str
    file_type: str
    document_id: str
    chunk_id: str
    chunk_index: int
    text: str
    char_count: int
    content_type: str = "document_chunk"
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    slide_number: int | None = None
    slide_title: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None

    def to_payload(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class SheetPayload:
    """Semantic discovery payload for one Excel/CSV sheet (spec section 4)."""

    user_id: str
    session_id: str
    file_id: str
    filename: str
    workbook_id: str
    sheet_id: str
    sheet_name: str
    table_name: str
    text: str
    content_type: str = "excel_sheet"

    def to_payload(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def _build_client(url: str, api_key: str | None) -> AsyncQdrantClient:
    # Server-only - no embedded/:memory: mode. config.py's
    # _require_qdrant_server validator rejects a non-http(s) URL at
    # startup with a clear error.
    return AsyncQdrantClient(url=url, api_key=api_key or None)


@lru_cache
def get_qdrant_client() -> AsyncQdrantClient:
    settings = get_settings()
    return _build_client(settings.qdrant_url, settings.qdrant_api_key)


class QdrantService:
    def __init__(self, client: AsyncQdrantClient | None = None, collection: str | None = None, dim: int | None = None) -> None:
        settings = get_settings()
        self._client = client or get_qdrant_client()
        self._collection = collection or settings.qdrant_collection
        self._dim = dim or settings.embedding_dim
        self._client_url_hint = settings.qdrant_url

    async def ensure_collection(self) -> None:
        """Called lazily on first real use (upsert/search/get_stats), not
        eagerly at app startup, so the "collection created" log line
        always corresponds to a real action, not just a process boot."""
        exists = await self._client.collection_exists(self._collection)
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(size=self._dim, distance=models.Distance.COSINE),
            )
            logger.info(
                "Qdrant collection '%s' newly created on server %s (dim=%s, 0 points - empty "
                "container, nothing uploaded yet).",
                self._collection, self._client_url_hint, self._dim,
            )
        else:
            info = await self._client.get_collection(self._collection)
            logger.info(
                "Qdrant collection '%s' already exists (dim=%s, %d point(s) stored)",
                self._collection, self._dim, info.points_count,
            )

    async def upsert(self, points: list[tuple[str, list[float], dict[str, Any]]]) -> None:
        if not points:
            return
        await self.ensure_collection()
        for point_id, vector, _payload in points:
            if len(vector) != self._dim:
                raise ValueError(
                    f"Embedding provider returned a {len(vector)}-dim vector for point "
                    f"{point_id}, but the Qdrant collection '{self._collection}' expects "
                    f"{self._dim} (from EMBEDDING_DIM). Set EMBEDDING_DIM to match your "
                    "embedding model's actual output size."
                )
        structs = [
            models.PointStruct(id=point_id, vector=vector, payload=payload)
            for point_id, vector, payload in points
        ]
        await self._client.upsert(collection_name=self._collection, points=structs)

    async def search(
        self,
        *,
        query_vector: list[float],
        user_id: str,
        session_id: str,
        content_type: str | None = None,
        file_id: str | None = None,
        top_k: int = 6,
    ) -> list[dict[str, Any]]:
        await self.ensure_collection()
        must: list[models.FieldCondition] = [
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
            models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id)),
        ]
        if content_type:
            must.append(models.FieldCondition(key="content_type", match=models.MatchValue(value=content_type)))
        if file_id:
            must.append(models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id)))

        result = await self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=models.Filter(must=must),
            limit=top_k,
            with_payload=True,
        )
        points = [{"score": p.score, "payload": p.payload} for p in result.points]

        if points:
            logger.info("=" * 88)
            logger.info(
                "[RETRIEVAL] %d candidate(s) from Qdrant (content_type=%s, top_k=%d, session=%s)",
                len(points), content_type, top_k, session_id,
            )
            for i, p in enumerate(points):
                logger.info("")
                logger.info("*" * 50)
                logger.info("")
                logger.info(
                    "[CANDIDATE %d/%d] score=%.4f | %s",
                    i + 1, len(points), p["score"], describe_point_location(p["payload"]),
                )
                logger.info(p["payload"].get("text") or "")
            logger.info("")
            logger.info("*" * 50)
            logger.info("=" * 88)

        return points

    async def describe_current_state(self) -> dict[str, Any]:
        """Read-only status check for startup logging - reports whatever is
        actually in Qdrant right now WITHOUT creating the collection as a
        side effect (unlike ensure_collection()/get_stats(), which create
        it if missing). Safe to call before any real upload/question has
        happened, so main.py's startup log can honestly say "nothing here
        yet" instead of the collection getting created (and logged as
        "newly created") purely as a byproduct of checking."""
        exists = await self._client.collection_exists(self._collection)
        if not exists:
            return {
                "collection": self._collection, "exists": False,
                "points_total": 0, "points_by_content_type": {},
            }
        stats = await self.get_stats()  # collection already confirmed to exist - no creation happens
        stats["exists"] = True
        return stats

    async def scroll_all_points(self, *, limit: int = 500) -> list[dict[str, Any]]:
        """Fetch every point currently stored (payload only, never the raw
        embedding vector - a wall of floats isn't useful to print). Paginates
        via Qdrant's scroll API so it works regardless of collection size,
        capped at `limit` total points. Used for startup visibility, so the
        assumption here is a local dev/demo-sized collection, not production
        scale - printing thousands of points on every boot wouldn't be
        readable anyway."""
        points: list[dict[str, Any]] = []
        offset = None
        while len(points) < limit:
            batch, next_offset = await self._client.scroll(
                collection_name=self._collection,
                limit=min(256, limit - len(points)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend({"id": p.id, "payload": p.payload} for p in batch)
            if next_offset is None:
                break
            offset = next_offset
        return points

    async def get_stats(self) -> dict[str, Any]:
        """Live snapshot of this collection - point counts overall and by
        content_type. Used by the /api/debug/state endpoint."""
        await self.ensure_collection()
        info = await self._client.get_collection(self._collection)
        by_type = {}
        for content_type in ("document_chunk", "excel_sheet"):
            count_result = await self._client.count(
                collection_name=self._collection,
                count_filter=models.Filter(
                    must=[models.FieldCondition(key="content_type", match=models.MatchValue(value=content_type))]
                ),
            )
            by_type[content_type] = count_result.count
        return {
            "collection": self._collection,
            "dim": self._dim,
            "points_total": info.points_count,
            "points_by_content_type": by_type,
        }

    async def delete_collection(self) -> None:
        """Drop this whole collection outright. Used by the test suite's
        session-scoped teardown (conftest.py) so test data doesn't
        accumulate across pytest runs on the real server. Safe only
        because tests use a dedicated QDRANT_COLLECTION name - NEVER call
        this against a collection real user data could be in."""
        exists = await self._client.collection_exists(self._collection)
        if exists:
            await self._client.delete_collection(self._collection)

    async def delete_session(self, *, user_id: str, session_id: str) -> None:
        """Delete every point belonging to this session (and, defensively,
        this user - session_id is already unique but this guards against a
        cross-user filter bug ever deleting the wrong data)."""
        exists = await self._client.collection_exists(self._collection)
        if not exists:
            return
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                        models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id)),
                    ]
                )
            ),
        )
        logger.info("Deleted Qdrant points for session %s", session_id)


@lru_cache
def get_qdrant_service() -> QdrantService:
    return QdrantService()
