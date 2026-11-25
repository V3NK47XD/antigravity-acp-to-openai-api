"""Unit tests for OpenAI models and Pydantic schemas."""

import pytest
from antigravity_proxy.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChunkChoice,
    ContentPartImage,
    ContentPartText,
    Delta,
    ImageUrl,
    ModelListResponse,
    ModelObject,
    ResponseMessage,
    UsageInfo,
)


def test_chat_message_plain_text():
    msg = ChatMessage(role="user", content="Hello Antigravity!")
    assert msg.role == "user"
    assert msg.get_text() == "Hello Antigravity!"


def test_chat_message_multimodal():
    msg = ChatMessage(
        role="user",
        content=[
            ContentPartText(text="Look at this picture:"),
            ContentPartImage(image_url=ImageUrl(url="data:image/png;base64,iVBORw0KGgo=")),
        ],
    )
    assert msg.role == "user"
    assert "Look at this picture:" in msg.get_text()
    assert len(msg.content) == 2


def test_chat_completion_request_defaults():
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="What is 2+2?")]
    )
    assert req.model == "gemini-3.8-flash-high"
    assert req.stream is False
    assert len(req.messages) == 1


def test_chat_completion_response_serialization():
    resp = ChatCompletionResponse(
        id="chatcmpl-test12345",
        model="gemini-3.8-flash-high",
        choices=[
            Choice(
                index=0,
                message=ResponseMessage(
                    role="assistant",
                    content="The answer is 4.",
                    reasoning_content="Simple arithmetic: 2 + 2 = 4.",
                ),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    data = resp.model_dump()
    assert data["id"] == "chatcmpl-test12345"
    assert data["choices"][0]["message"]["content"] == "The answer is 4."
    assert data["choices"][0]["message"]["reasoning_content"] == "Simple arithmetic: 2 + 2 = 4."
    assert data["usage"]["total_tokens"] == 15


def test_chat_completion_chunk_serialization():
    chunk = ChatCompletionChunk(
        id="chatcmpl-chunk-1",
        model="claude-sonnet-4-6",
        choices=[
            ChunkChoice(
                index=0,
                delta=Delta(content="Hello"),
                finish_reason=None,
            )
        ],
    )
    json_str = chunk.model_dump_json()
    assert "chat.completion.chunk" in json_str
    assert "Hello" in json_str


def test_model_list_response():
    resp = ModelListResponse(
        object="list",
        data=[
            ModelObject(id="gemini-3.8-flash-high", owned_by="antigravity"),
            ModelObject(id="claude-sonnet-4-6", owned_by="antigravity"),
        ],
    )
    assert len(resp.data) == 2
    assert resp.data[0].id == "gemini-3.8-flash-high"
