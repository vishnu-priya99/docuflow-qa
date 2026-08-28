from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_current_user_id,
    get_db,
    get_embedder,
    get_llm,
    get_qdrant,
    get_readonly_db,
    get_reranker,
)
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.graph.deps import GraphDeps
from app.graph.workflow import run_workflow
from app.models.message import Message
from app.schemas.chat import ChatRequest, ChatResponse
from app.services import session_service
from app.services.embeddings.base import EmbeddingProvider
from app.services.llm.base import LLMProvider
from app.services.reranking.base import RerankProvider
from app.vector.qdrant_client import QdrantService

router = APIRouter()
logger = get_logger(__name__)


@router.post("/{session_id}/chat", response_model=ChatResponse)
async def ask_question(
    session_id: str,
    payload: ChatRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
    readonly_db: AsyncSession = Depends(get_readonly_db),
    qdrant: QdrantService = Depends(get_qdrant),
    embedder: EmbeddingProvider = Depends(get_embedder),
    llm: LLMProvider = Depends(get_llm),
    reranker: RerankProvider = Depends(get_reranker),
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    session = await session_service.get_session(db, user_id=user_id, session_id=session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    logger.info("=" * 88)
    logger.info('[QUESTION] user=%s session=%s: "%s"', user_id, session_id, payload.question)

    user_message = Message(session_id=session_id, user_id=user_id, role="user", content=payload.question)
    db.add(user_message)
    await db.flush()

    deps = GraphDeps(
        db=db, readonly_db=readonly_db, qdrant=qdrant, embedder=embedder, llm=llm,
        reranker=reranker, settings=settings,
    )
    result = await run_workflow(question=payload.question, user_id=user_id, session_id=session_id, deps=deps)

    answer = result.get("answer") or "I couldn't find that information in the uploaded files."
    question_type = result.get("question_type", "SEMANTIC")
    sources = result.get("sources", [])

    logger.info('[ANSWER] (%s) "%s"', question_type, answer)
    logger.info("[SOURCES] %d source(s) returned", len(sources))
    for idx, s in enumerate(sources):
        logger.info("")
        logger.info("*" * 50)
        logger.info("")
        logger.info("[SOURCE %d/%d]", idx + 1, len(sources))
        logger.info("%s", s)
    logger.info("")
    logger.info("*" * 50)
    logger.info("=" * 88)

    assistant_message = Message(
        session_id=session_id,
        user_id=user_id,
        role="assistant",
        content=answer,
        question_type=question_type,
        sources=sources,
    )
    db.add(assistant_message)
    await db.commit()

    return ChatResponse(
        message_id=assistant_message.message_id,
        question=payload.question,
        answer=answer,
        question_type=question_type,
        sources=sources,
    )
