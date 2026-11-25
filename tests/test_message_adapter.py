"""Unit tests for message adapter and prompt translation."""

import pytest
from antigravity_proxy.message_adapter import (
    convert_messages_to_prompt_blocks,
    get_acp_model_config,
    parse_image_to_acp_block,
    resolve_model_name,
)
from antigravity_proxy.models import ChatMessage, ContentPartImage, ContentPartText, ImageUrl


def test_resolve_model_name():
    assert resolve_model_name("gpt-4o") == "gemini-3.8-flash-high"
    assert resolve_model_name("claude-3-7-sonnet") == "claude-sonnet-4-6"
    assert resolve_model_name("claude-opus") == "claude-opus-4-6-thinking"
    assert resolve_model_name("gpt-oss") == "gpt-oss-120b-medium"
    assert resolve_model_name("unknown-model") == "unknown-model"
    assert resolve_model_name(None) == "gemini-3.8-flash-high"


def test_get_acp_model_config():
    acp_m, thought = get_acp_model_config("gemini-3.8-flash-high")
    assert acp_m == "gemini-3.8-flash"
    assert thought == "high"

    acp_m, thought = get_acp_model_config("gemini-3.8-flash-low")
    assert acp_m == "gemini-3.8-flash"
    assert thought == "low"

    acp_m, thought = get_acp_model_config("claude-sonnet-4-6")
    assert acp_m == "claude-sonnet-4-6"
    assert thought is None


@pytest.mark.asyncio
async def test_convert_single_user_message():
    messages = [
        ChatMessage(role="user", content="Hello Antigravity!"),
    ]
    sys_prompt, blocks = await convert_messages_to_prompt_blocks(messages, is_resumed_session=False)
    assert sys_prompt is None
    assert len(blocks) == 1
    assert blocks[0]["type"] == "text"
    assert blocks[0]["text"] == "Hello Antigravity!"


@pytest.mark.asyncio
async def test_convert_system_and_user_message():
    messages = [
        ChatMessage(role="system", content="You are a helpful coding assistant."),
        ChatMessage(role="user", content="Write a sort function."),
    ]
    sys_prompt, blocks = await convert_messages_to_prompt_blocks(messages, is_resumed_session=False)
    assert sys_prompt == "You are a helpful coding assistant."
    assert any("Write a sort function." in b["text"] for b in blocks)


@pytest.mark.asyncio
async def test_convert_multiturn_new_session():
    messages = [
        ChatMessage(role="system", content="Be concise."),
        ChatMessage(role="user", content="Hi!"),
        ChatMessage(role="assistant", content="Hello!"),
        ChatMessage(role="user", content="Tell me a joke."),
    ]
    sys_prompt, blocks = await convert_messages_to_prompt_blocks(messages, is_resumed_session=False)
    assert sys_prompt == "Be concise."
    # The prompt should contain context of previous conversation and the latest turn
    context_text = blocks[0]["text"]
    assert "User: Hi!" in context_text
    assert "Assistant: Hello!" in context_text
    assert "Tell me a joke." in blocks[-1]["text"]


@pytest.mark.asyncio
async def test_convert_resumed_session_only_sends_latest():
    messages = [
        ChatMessage(role="user", content="Turn 1"),
        ChatMessage(role="assistant", content="Response 1"),
        ChatMessage(role="user", content="Turn 2"),
    ]
    sys_prompt, blocks = await convert_messages_to_prompt_blocks(messages, is_resumed_session=True)
    assert len(blocks) == 1
    assert blocks[0]["text"] == "Turn 2"


@pytest.mark.asyncio
async def test_parse_data_uri_image():
    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    block = await parse_image_to_acp_block(data_uri)
    assert block["type"] == "image"
    assert block["mimeType"] == "image/png"
    assert block["data"].startswith("iVBORw0KGgo")
