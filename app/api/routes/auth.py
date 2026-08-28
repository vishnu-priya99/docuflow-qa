from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.schemas.user import LoginRequest, LoginResponse
from app.services.auth_service import get_or_create_user

router = APIRouter()


@router.post("/login", response_model=LoginResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)) -> LoginResponse:
    user = await get_or_create_user(db, payload.user_id)
    await db.commit()
    return LoginResponse(user_id=user.user_id)
