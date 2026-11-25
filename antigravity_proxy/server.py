"""FastAPI server exposing OpenAI-compatible endpoints over Antigravity ACP."""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from antigravity_proxy.acp_client import ACPClient, ACPError
from antigravity_proxy.config import settings
from antigravity_proxy.message_adapter import (
    MODEL_ALIASES,
    convert_messages_to_prompt_blocks,
    resolve_model_name,
)
from antigravity_proxy.models import (
    APIErrorDetail,
    APIErrorResponse,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChunkChoice,
    CompletionTokensDetails,
    Delta,
    ModelListResponse,
    ModelObject,
    PromptTokensDetails,
    ResponseMessage,
    UsageInfo,
)
from antigravity_proxy.session_manager import SessionManager

logger = logging.getLogger("antigravity_proxy.server")

# Global client and session manager instances
acp_client: ACPClient | None = None
session_manager: SessionManager | None = None

# ---------------------------------------------------------------------------
# In-Memory Heavy Model Cache
# Populated once on server restart, never queried live during requests.
# ---------------------------------------------------------------------------
cached_models_response: ModelListResponse = ModelListResponse(object="list", data=[])
cached_models_dict: dict[str, ModelObject] = {}


def _build_model_cache(model_id_set: set[str]) -> None:
    """Build immutable in-memory model response and lookup dict."""
    global cached_models_response, cached_models_dict
    created_time = 1700000000
    model_objects = [
        ModelObject(
            id=mid,
            created=created_time,
            owned_by="antigravity",
        )
        for mid in sorted(model_id_set)
        if mid
    ]
    cached_models_response = ModelListResponse(object="list", data=model_objects)
    cached_models_dict = {m.id: m for m in model_objects}


# Pre-seed fallback cache at module load
_default_ids = set([
    "gemini-3.8-flash-high",
    "gemini-3.8-flash-medium",
    "gemini-3.8-flash-low",
    "claude-sonnet-4-6",
    "claude-opus-4-6-thinking",
    "gpt-oss-120b-medium",
] + list(MODEL_ALIASES.keys()))
_build_model_cache(_default_ids)


async def init_models_cache(client: ACPClient) -> None:
    """Check models from Antigravity ACP once on server startup and store in memory."""
    logger.info("Initializing in-memory models cache from Antigravity ACP (server boot)...")
    model_ids: set[str] = set()

    try:
        session_res = await client.session_new(cwd=settings.workspace_dir)
        models_data = session_res.get("models", {}).get("availableModels", [])
        for m in models_data:
            mid = m.get("modelId")
            if mid:
                model_ids.add(mid)

        for opt in session_res.get("configOptions", []):
            if opt.get("id") == "model":
                for choice in opt.get("options", []):
                    val = choice.get("value")
                    if val:
                        model_ids.add(val)

        logger.info(
            "Discovered %d native models from Antigravity ACP: %s",
            len(model_ids),
            sorted(model_ids),
        )
    except Exception as e:
        logger.warning(
            "Could not query Antigravity ACP for models during startup (%s). Using defaults.", e
        )
        model_ids.update([
            "gemini-3.8-flash-high",
            "gemini-3.8-flash-medium",
            "gemini-3.8-flash-low",
            "claude-sonnet-4-6",
            "claude-opus-4-6-thinking",
            "gpt-oss-120b-medium",
        ])

    # Include convenient aliases
    model_ids.update(list(MODEL_ALIASES.keys()))

    _build_model_cache(model_ids)
    logger.info(
        "Heavy model cache initialized with %d models. Live lookups are disabled.",
        len(cached_models_dict),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager starting ACP subprocess on boot and terminating on shutdown."""
    global acp_client, session_manager
    logger.info("Starting Antigravity ACP client...")
    acp_client = ACPClient(
        executable=settings.agy_bin,
        extra_args=settings.agy_extra_args,
        workspace_dir=settings.workspace_dir,
    )
    await acp_client.start()
    session_manager = SessionManager(acp_client)

    # Heavy cache models once during server boot
    await init_models_cache(acp_client)

    logger.info("Proxy server ready for OpenAI requests.")

    yield

    logger.info("Shutting down Antigravity ACP client...")
    if acp_client:
        await acp_client.close()


app = FastAPI(
    title="Antigravity ACP OpenAI Proxy",
    description="High-performance OpenAI-compatible proxy powered by Google Antigravity Agent Client Protocol (ACP)",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for web apps (LibreChat, Chatbox, Cherry Studio, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-ID"],
)


# ---------------------------------------------------------------------------
# Authentication Dependency
# ---------------------------------------------------------------------------

async def verify_api_key(authorization: str | None = Header(default=None)) -> None:
    """Validate Bearer token if API key is configured."""
    if not settings.api_key:
        return

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"message": "Missing Authorization header", "type": "invalid_request_error"}},
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"message": "Invalid API key provided", "type": "invalid_request_error"}},
        )


# ---------------------------------------------------------------------------
# Health / Info Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
async def health_check():
    """Proxy health check and status."""
    return {
        "status": "healthy",
        "service": "antigravity-acp-openai-proxy",
        "version": "1.0.0",
        "acp": {
            "connected": acp_client is not None and acp_client._is_running,
            "server_info": acp_client.server_info if acp_client else {},
            "active_sessions": session_manager.get_active_session_count() if session_manager else 0,
            "cached_models_count": len(cached_models_dict),
        },
    }


# ---------------------------------------------------------------------------
# Models Endpoints (Served exclusively from in-memory heavy cache)
# ---------------------------------------------------------------------------

@app.get("/v1/models", tags=["Models"], dependencies=[Depends(verify_api_key)])
@app.get("/models", tags=["Models"], dependencies=[Depends(verify_api_key)])
async def list_models() -> ModelListResponse:
    """Return heavily cached models list directly from memory (never live-queried)."""
    return cached_models_response


@app.get("/v1/models/{model_id}", tags=["Models"], dependencies=[Depends(verify_api_key)])
async def get_model(model_id: str) -> ModelObject:
    """Retrieve details for a specific model from in-memory cache."""
    model = cached_models_dict.get(model_id)
    if model:
        return model

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": {"message": f"Model '{model_id}' not found in cached models", "type": "invalid_request_error"}},
    )


# ---------------------------------------------------------------------------
# Chat Completions Endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/v1/chat/completions",
    tags=["Chat Completions"],
    dependencies=[Depends(verify_api_key)],
    responses={
        200: {"description": "Successful completion"},
        400: {"model": APIErrorResponse},
        401: {"model": APIErrorResponse},
        500: {"model": APIErrorResponse},
    },
)
@app.post(
    "/chat/completions",
    tags=["Chat Completions"],
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
async def chat_completions(
    request: ChatCompletionRequest,
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
):
    """Handle OpenAI-compatible chat completion request (streaming or non-streaming)."""
    assert session_manager is not None
    assert acp_client is not None

    # Determine requested session ID if caller passed one
    candidate_session_id = request.session_id or request.conversation_id or x_session_id
    resolved_model = resolve_model_name(request.model, settings.default_model)

    try:
        # Check if this candidate session is already active
        is_already_active = bool(
            candidate_session_id and candidate_session_id in session_manager._sessions
        )

        # Convert messages into system prompt and ACP prompt blocks
        system_prompt, prompt_blocks = await convert_messages_to_prompt_blocks(
            request.messages,
            is_resumed_session=is_already_active,
        )

        # Retrieve or instantiate ACP session
        session_id, is_resumed = await session_manager.get_or_create_session(
            session_id=candidate_session_id,
            requested_model=resolved_model,
            system_prompt=system_prompt,
        )

    except ACPError as e:
        logger.error("ACP Error during session preparation: %s", e)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": {"message": str(e), "type": "acp_error", "code": e.code}},
        )
    except Exception as e:
        logger.exception("Unexpected error setting up ACP session: %s", e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"message": str(e), "type": "internal_error"}},
        )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())

    # -----------------------------------------------------------------------
    # Streaming Response (Server-Sent Events)
    # -----------------------------------------------------------------------
    if request.stream:

        async def sse_event_stream() -> AsyncGenerator[str, None]:
            # Initial role delta chunk
            initial_chunk = ChatCompletionChunk(
                id=completion_id,
                created=created_time,
                model=resolved_model,
                choices=[ChunkChoice(index=0, delta=Delta(role="assistant"))],
            )
            yield f"data: {initial_chunk.model_dump_json()}\n\n"

            accumulated_usage = UsageInfo()

            try:
                async for event in acp_client.session_prompt_stream(session_id, prompt_blocks):
                    if event["type"] == "update":
                        update = event["data"]
                        update_type = update.get("sessionUpdate")

                        if update_type == "agent_message_chunk":
                            text = update.get("content", {}).get("text", "")
                            if text:
                                chunk = ChatCompletionChunk(
                                    id=completion_id,
                                    created=created_time,
                                    model=resolved_model,
                                    choices=[ChunkChoice(index=0, delta=Delta(content=text))],
                                )
                                yield f"data: {chunk.model_dump_json()}\n\n"

                        elif update_type == "agent_thought_chunk":
                            content_obj = update.get("content", {})
                            thought_text = (
                                content_obj.get("text", "")
                                if isinstance(content_obj, dict)
                                else str(content_obj)
                            )
                            if thought_text:
                                chunk = ChatCompletionChunk(
                                    id=completion_id,
                                    created=created_time,
                                    model=resolved_model,
                                    choices=[
                                        ChunkChoice(
                                            index=0,
                                            delta=Delta(reasoning_content=thought_text),
                                        )
                                    ],
                                )
                                yield f"data: {chunk.model_dump_json()}\n\n"

                    elif event["type"] == "final":
                        final_res = event.get("data", {})
                        usage_dict = final_res.get("usage", {})
                        if usage_dict:
                            accumulated_usage = UsageInfo(
                                prompt_tokens=usage_dict.get("inputTokens", 0),
                                completion_tokens=usage_dict.get("outputTokens", 0),
                                total_tokens=usage_dict.get("totalTokens", 0),
                                prompt_tokens_details=PromptTokensDetails(
                                    cached_tokens=usage_dict.get("cachedReadTokens", 0)
                                ),
                                completion_tokens_details=CompletionTokensDetails(
                                    reasoning_tokens=usage_dict.get("thoughtTokens", 0)
                                ),
                            )

                # Final finish chunk
                final_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created_time,
                    model=resolved_model,
                    choices=[ChunkChoice(index=0, delta=Delta(), finish_reason="stop")],
                    usage=accumulated_usage,
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as ex:
                logger.error("Error during streaming prompt: %s", ex)
                err_payload = {"error": {"message": str(ex), "type": "stream_error"}}
                yield f"data: {json.dumps(err_payload)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            sse_event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Session-ID": session_id,
            },
        )

    # -----------------------------------------------------------------------
    # Non-Streaming Response
    # -----------------------------------------------------------------------
    content_pieces: list[str] = []
    thought_pieces: list[str] = []
    usage_info = UsageInfo()

    try:
        async for event in acp_client.session_prompt_stream(session_id, prompt_blocks):
            if event["type"] == "update":
                update = event["data"]
                update_type = update.get("sessionUpdate")

                if update_type == "agent_message_chunk":
                    text = update.get("content", {}).get("text", "")
                    if text:
                        content_pieces.append(text)
                elif update_type == "agent_thought_chunk":
                    content_obj = update.get("content", {})
                    thought_text = (
                        content_obj.get("text", "")
                        if isinstance(content_obj, dict)
                        else str(content_obj)
                    )
                    if thought_text:
                        thought_pieces.append(thought_text)

            elif event["type"] == "final":
                final_res = event.get("data", {})
                u = final_res.get("usage", {})
                if u:
                    usage_info = UsageInfo(
                        prompt_tokens=u.get("inputTokens", 0),
                        completion_tokens=u.get("outputTokens", 0),
                        total_tokens=u.get("totalTokens", 0),
                        prompt_tokens_details=PromptTokensDetails(
                            cached_tokens=u.get("cachedReadTokens", 0)
                        ),
                        completion_tokens_details=CompletionTokensDetails(
                            reasoning_tokens=u.get("thoughtTokens", 0)
                        ),
                    )

        full_content = "".join(content_pieces)
        full_thought = "".join(thought_pieces) if thought_pieces else None

        response = ChatCompletionResponse(
            id=completion_id,
            created=created_time,
            model=resolved_model,
            choices=[
                Choice(
                    index=0,
                    message=ResponseMessage(
                        role="assistant",
                        content=full_content,
                        reasoning_content=full_thought,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=usage_info,
        )

        return Response(
            content=response.model_dump_json(),
            media_type="application/json",
            headers={"X-Session-ID": session_id},
        )

    except ACPError as e:
        logger.error("ACP execution error: %s", e)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": {"message": str(e), "type": "acp_error", "code": e.code}},
        )
    except Exception as e:
        logger.exception("Unexpected error executing prompt: %s", e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"message": str(e), "type": "server_error"}},
        )
