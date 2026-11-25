"""Pydantic data models for OpenAI API compatibility."""

import time
from typing import Any, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Multimodal Content Parts
# ---------------------------------------------------------------------------

class ImageUrl(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] = "auto"


class ContentPartText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ContentPartImage(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


ContentPart = ContentPartText | ContentPartImage


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function", "tool", "developer"]
    content: str | list[ContentPart] | None = None
    name: str | None = None

    def get_text(self) -> str:
        """Extract all plain text from content."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        texts: list[str] = []
        for part in self.content:
            if isinstance(part, ContentPartText) or (isinstance(part, dict) and part.get("type") == "text"):
                texts.append(part.text if isinstance(part, ContentPartText) else part.get("text", ""))
        return "\n".join(texts)


# ---------------------------------------------------------------------------
# Chat Completion Request
# ---------------------------------------------------------------------------

class ChatCompletionRequest(BaseModel):
    model: str = "gemini-3.8-flash-high"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = 1
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None

    # Custom extensions for session continuity
    session_id: str | None = Field(default=None, description="ACP Session ID for multi-turn continuity")
    conversation_id: str | None = Field(default=None, description="Alias for session_id")


# ---------------------------------------------------------------------------
# Usage Information
# ---------------------------------------------------------------------------

class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int = 0


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: PromptTokensDetails | None = None
    completion_tokens_details: CompletionTokensDetails | None = None


# ---------------------------------------------------------------------------
# Non-Streaming Chat Completion Response
# ---------------------------------------------------------------------------

class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    reasoning_content: str | None = None


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: Literal["stop", "length", "content_filter"] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: UsageInfo | None = None
    system_fingerprint: str | None = None


# ---------------------------------------------------------------------------
# Streaming Chat Completion Chunk
# ---------------------------------------------------------------------------

class Delta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None
    reasoning_content: str | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: Delta
    finish_reason: Literal["stop", "length", "content_filter"] | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChunkChoice]
    usage: UsageInfo | None = None


# ---------------------------------------------------------------------------
# Models List Response
# ---------------------------------------------------------------------------

class ModelPermission(BaseModel):
    id: str = "modelperm-default"
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: None = None
    is_blocking: bool = False


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "antigravity"
    permission: list[ModelPermission] = Field(default_factory=lambda: [ModelPermission()])
    root: str | None = None
    parent: str | None = None


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]


# ---------------------------------------------------------------------------
# Error Response
# ---------------------------------------------------------------------------

class APIErrorDetail(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | int | None = None


class APIErrorResponse(BaseModel):
    error: APIErrorDetail
