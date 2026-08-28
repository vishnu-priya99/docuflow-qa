from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User


async def get_or_create_user(db: AsyncSession, user_id: str) -> User:
    user_id = user_id.strip()
    if not user_id:
        raise ValueError("user_id must not be empty")

    result = await db.execute(select(User).where(User.user_id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(user_id=user_id)
        db.add(user)
        await db.flush()
    return user
