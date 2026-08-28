from fastapi import APIRouter

from app.api.routes import auth, chat, debug, files, health, sessions

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
api_router.include_router(files.router, prefix="/sessions", tags=["files"])
api_router.include_router(chat.router, prefix="/sessions", tags=["chat"])
api_router.include_router(debug.router, prefix="/debug", tags=["debug"])
