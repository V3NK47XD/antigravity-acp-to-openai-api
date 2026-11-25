"""Adapter converting OpenAI messages to ACP prompt blocks and managing model mapping."""

import base64
import logging
import re
from typing import Any
import httpx
from antigravity_proxy.models import ChatMessage, ContentPartImage, ContentPartText

logger = logging.getLogger("antigravity_proxy.adapter")

# Model aliases mapping common OpenAI / Anthropic model identifiers to Antigravity models
MODEL_ALIASES: dict[str, str] = {
    # Gemini
    "gemini": "gemini-3.8-flash-high",
    "gemini-flash": "gemini-3.8-flash-high",
    "gemini-3.8-flash": "gemini-3.8-flash-high",
    "gemini-3.8": "gemini-3.8-flash-high",
    "gemini-2.5-flash": "gemini-3.8-flash-high",
    "gemini-2.5-pro": "gemini-3.8-flash-high",
    # Claude
    "claude-sonnet": "claude-sonnet-4-6",
    "claude-3-7-sonnet": "claude-sonnet-4-6",
    "claude-3-5-sonnet": "claude-sonnet-4-6",
    "claude-opus": "claude-opus-4-6-thinking",
    "claude-3-opus": "claude-opus-4-6-thinking",
    # OpenAI
    "gpt-4o": "gemini-3.8-flash-high",
    "gpt-4o-mini": "gemini-3.8-flash-low",
    "gpt-4": "gemini-3.8-flash-high",
    "gpt-3.5-turbo": "gemini-3.8-flash-low",
    "o1": "claude-sonnet-4-6",
    "o3-mini": "gemini-3.8-flash-high",
    "gpt-oss": "gpt-oss-120b-medium",
    "default": "gemini-3.8-flash-high",
}


def resolve_model_name(requested_model: str | None, default_model: str = "gemini-3.8-flash-high") -> str:
    """Normalize model identifier using alias mapping."""
    if not requested_model:
        return default_model

    clean_name = requested_model.strip().lower()
    if clean_name in MODEL_ALIASES:
        return MODEL_ALIASES[clean_name]

    return requested_model


def get_acp_model_config(resolved_model: str) -> tuple[str, str | None]:
    """
    Deconstruct resolved model identifier into ACP configOption ('model' and optional 'thought_level').
    Returns (acp_model_value, thought_level_value).
    """
    clean = resolved_model.strip().lower()
    if clean == "gemini-3.8-flash-high":
        return "gemini-3.8-flash", "high"
    elif clean == "gemini-3.8-flash-medium":
        return "gemini-3.8-flash", "medium"
    elif clean == "gemini-3.8-flash-low":
        return "gemini-3.8-flash", "low"
    elif clean.startswith("claude-sonnet"):
        return "claude-sonnet-4-6", None
    elif clean.startswith("claude-opus"):
        return "claude-opus-4-6-thinking", None
    elif "gpt-oss" in clean:
        return "gpt-oss-120b-medium", None
    else:
        # Fallback
        return clean, None


async def parse_image_to_acp_block(url: str) -> dict[str, str]:
    """Convert data URI or fetch remote image into an ACP image block."""
    data_uri_match = re.match(r"^data:([^;]+);base64,(.+)$", url, re.DOTALL)
    if data_uri_match:
        mime_type = data_uri_match.group(1)
        b64_data = data_uri_match.group(2).strip()
        return {
            "type": "image",
            "mimeType": mime_type,
            "data": b64_data,
        }

    # If it's an HTTP URL, attempt async download
    if url.startswith("http://") or url.startswith("https://"):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                mime_type = resp.headers.get("content-type", "image/png").split(";")[0]
                b64_data = base64.b64encode(resp.content).decode("ascii")
                return {
                    "type": "image",
                    "mimeType": mime_type,
                    "data": b64_data,
                }
        except Exception as e:
            logger.warning("Failed to fetch image URL %s: %s", url, e)
            return {"type": "text", "text": f"[Image at {url} could not be loaded]"}

    # Fallback: treat as plain text reference
    return {"type": "text", "text": f"[Image: {url}]"}


async def convert_messages_to_prompt_blocks(
    messages: list[ChatMessage],
    is_resumed_session: bool = False,
) -> tuple[str | None, list[dict[str, Any]]]:
    """
    Extract system instructions and build ACP prompt blocks.

    If is_resumed_session is True, only the final user turn is transformed into
    prompt blocks since prior context is already in the ACP session.

    If is_resumed_session is False, system instructions and any conversation
    history are composed appropriately for the first prompt turn.
    """
    system_instructions: list[str] = []
    user_and_assistant_messages: list[ChatMessage] = []

    for msg in messages:
        if msg.role in ("system", "developer"):
            text = msg.get_text().strip()
            if text:
                system_instructions.append(text)
        else:
            user_and_assistant_messages.append(msg)

    combined_system_prompt = "\n\n".join(system_instructions) if system_instructions else None

    # Case 1: Session is resumed and already contains previous turns
    if is_resumed_session:
        # Find the latest user message
        if not user_and_assistant_messages:
            return combined_system_prompt, [{"type": "text", "text": ""}]

        latest_msg = user_and_assistant_messages[-1]
        blocks = await _message_to_blocks(latest_msg)
        return combined_system_prompt, blocks

    # Case 2: Fresh session, but we have multiple turns in request history
    if len(user_and_assistant_messages) <= 1:
        # Single turn or no conversation history
        if not user_and_assistant_messages:
            blocks = [{"type": "text", "text": "Hello"}]
        else:
            blocks = await _message_to_blocks(user_and_assistant_messages[0])

        # If there are system instructions, prepend them to prompt blocks if needed
        if combined_system_prompt:
            # We also pass systemPrompt to session/new, but adding context ensures strict compliance
            blocks.insert(0, {"type": "text", "text": f"[System Instructions]\n{combined_system_prompt}\n\n"})
        return combined_system_prompt, blocks

    # Multi-turn history provided to a new session
    # Format the prior turns into context, with the latest turn as active
    history_lines: list[str] = []
    if combined_system_prompt:
        history_lines.append(f"[System Instructions]\n{combined_system_prompt}\n")

    history_lines.append("[Previous Conversation]")
    for msg in user_and_assistant_messages[:-1]:
        role_label = "User" if msg.role == "user" else "Assistant"
        content_text = msg.get_text()
        history_lines.append(f"{role_label}: {content_text}")

    history_lines.append("\n[Current Turn]")
    history_header = "\n".join(history_lines) + "\n"

    latest_msg = user_and_assistant_messages[-1]
    latest_blocks = await _message_to_blocks(latest_msg)

    # Prepend history header block
    prompt_blocks = [{"type": "text", "text": history_header}] + latest_blocks
    return combined_system_prompt, prompt_blocks


async def _message_to_blocks(msg: ChatMessage) -> list[dict[str, Any]]:
    """Convert a single ChatMessage into ACP content blocks."""
    blocks: list[dict[str, Any]] = []

    if msg.content is None:
        return [{"type": "text", "text": ""}]

    if isinstance(msg.content, str):
        return [{"type": "text", "text": msg.content}]

    # Multimodal parts
    for part in msg.content:
        if isinstance(part, ContentPartText):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ContentPartImage):
            block = await parse_image_to_acp_block(part.image_url.url)
            blocks.append(block)
        elif isinstance(part, dict):
            ptype = part.get("type")
            if ptype == "text":
                blocks.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "image_url":
                img_url_obj = part.get("image_url", {})
                url = img_url_obj.get("url", "") if isinstance(img_url_obj, dict) else str(img_url_obj)
                block = await parse_image_to_acp_block(url)
                blocks.append(block)

    if not blocks:
        blocks = [{"type": "text", "text": ""}]
    return blocks
