from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    message_id: str
    question: str
    answer: str
    question_type: str
    sources: list[dict[str, Any]] = Field(default_factory=list)


class MessageOut(BaseModel):
    message_id: str
    role: str
    content: str
    question_type: str | None = None
    sources: list[dict[str, Any]] | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageListOut(BaseModel):
    messages: list[MessageOut]
