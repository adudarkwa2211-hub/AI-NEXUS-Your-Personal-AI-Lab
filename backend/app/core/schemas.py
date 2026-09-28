from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message must not be empty")
        return cleaned


class SourceChunk(BaseModel):
    id: int
    source: str
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    llm_model: str
    index_loaded: bool
    index_size: int
