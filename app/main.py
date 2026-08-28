from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.warmup import warm_up_models
from app.db.init_db import init_models
from app.vector.qdrant_client import describe_point_location, get_qdrant_service

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_models()

    # Read-only status report - does not create the collection as a side
    # effect. Collection provisioning happens lazily on first real use
    # (see QdrantService.ensure_collection()).
    qdrant_state = await get_qdrant_service().describe_current_state()
    if qdrant_state["exists"]:
        logger.info(
            "Qdrant collection '%s' status: EXISTS - %d point(s) total %s",
            qdrant_state["collection"], qdrant_state["points_total"], qdrant_state["points_by_content_type"],
        )
        if qdrant_state["points_total"] > 0:
            points = await get_qdrant_service().scroll_all_points()
            for i, p in enumerate(points):
                logger.info("")
                logger.info("*" * 50)
                logger.info("")
                logger.info(
                    "[EXISTING POINT %d/%d] id=%s | %s",
                    i + 1, len(points), p["id"], describe_point_location(p["payload"]),
                )
                logger.info(p["payload"].get("text") or "")
            logger.info("")
            logger.info("*" * 50)
    else:
        logger.info(
            "Qdrant collection '%s' status: DOES NOT EXIST YET - 0 points. "
            "Will be created automatically on first upload/question.",
            settings.qdrant_collection,
        )

    await warm_up_models()

    logger.info("Startup complete (llm=%s, embeddings=%s, storage=%s)", settings.llm_provider, settings.embedding_provider, settings.storage_backend)
    yield


app = FastAPI(title="Document Q&A System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
