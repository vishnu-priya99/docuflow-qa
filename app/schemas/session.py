from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    title: str = Field(default="New chat", max_length=255)


class SessionOut(BaseModel):
    session_id: str
    user_id: str
    title: str
    created_at: datetime

    model_config = {"from_attributes": True}


class SessionListOut(BaseModel):
    sessions: list[SessionOut]
