"""Chat + generation schemas (OpenAI-compatible shape where practical)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel

Role = Literal["system", "user", "assistant"]


class SamplingParams(BaseModel):
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=1000)
    max_tokens: int = Field(default=512, ge=1, le=131_072)
    repetition_penalty: float = Field(default=1.1, ge=0.5, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    stop: list[str] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: Role
    content: str


class ChatCompletionRequest(BaseModel):
    model_id: str
    messages: list[ChatMessage]
    system_prompt: str | None = None
    params: SamplingParams = Field(default_factory=SamplingParams)
    stream: bool = True
    conversation_id: str | None = None
    persist: bool = True


class GenerationUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_sec: float = 0.0
    latency_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    finish_reason: str = "stop"
    provenance: str = "simulated"
    engine: str = "demo"


class ChatCompletionResponse(BaseModel):
    id: str
    conversation_id: str | None = None
    model_id: str
    content: str
    usage: GenerationUsage
    created_at: datetime


class MessageRead(ORMModel):
    id: str
    conversation_id: str
    role: str
    content: str
    model_id: str | None = None
    provenance: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tokens_per_sec: float | None = None
    latency_ms: float | None = None
    time_to_first_token_ms: float | None = None
    finish_reason: str | None = None
    error: str | None = None
    created_at: datetime


class ConversationRead(ORMModel):
    id: str
    title: str
    model_id: str | None = None
    system_prompt: str
    params: dict[str, Any]
    pinned: bool
    is_demo: bool
    created_at: datetime
    updated_at: datetime


class ConversationDetail(ConversationRead):
    messages: list[MessageRead] = Field(default_factory=list)


class ConversationCreate(BaseModel):
    title: str | None = None
    model_id: str | None = None
    system_prompt: str = ""
    params: SamplingParams = Field(default_factory=SamplingParams)


class ConversationUpdate(BaseModel):
    title: str | None = None
    model_id: str | None = None
    system_prompt: str | None = None
    params: SamplingParams | None = None
    pinned: bool | None = None


class MessageUpdate(BaseModel):
    content: str


class TokenCountRequest(BaseModel):
    model_id: str | None = None
    text: str


class TokenCountResponse(BaseModel):
    tokens: int
    characters: int
    method: Literal["tokenizer", "estimated"]
