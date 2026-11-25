"""Session manager for caching, reusing, and pruning ACP sessions."""

import asyncio
import logging
import time
from dataclasses import dataclass
from antigravity_proxy.acp_client import ACPClient
from antigravity_proxy.config import settings
from antigravity_proxy.message_adapter import get_acp_model_config, resolve_model_name

logger = logging.getLogger("antigravity_proxy.session")


@dataclass
class ACPSessionInfo:
    session_id: str
    created_at: float
    last_accessed: float
    current_model: str
    current_thought_level: str | None
    turn_count: int = 0


class SessionManager:
    """Manages pool of active ACP sessions, lifecycle timeouts, and model configuration."""

    def __init__(self, acp_client: ACPClient):
        self.client = acp_client
        self._sessions: dict[str, ACPSessionInfo] = {}
        self._lock = asyncio.Lock()

    async def get_or_create_session(
        self,
        session_id: str | None = None,
        requested_model: str = "gemini-3.8-flash-high",
        system_prompt: str | None = None,
    ) -> tuple[str, bool]:
        """
        Retrieve an active session or create a new one.
        Returns a tuple of (session_id, is_resumed).
        """
        resolved_model = resolve_model_name(requested_model, settings.default_model)
        now = time.time()

        async with self._lock:
            # 1. Try to reuse existing active session
            if session_id and session_id in self._sessions:
                sess_info = self._sessions[session_id]
                # Check expiration
                if (now - sess_info.last_accessed) < settings.session_ttl_seconds:
                    sess_info.last_accessed = now
                    sess_info.turn_count += 1

                    # Switch model if requested model has changed
                    if sess_info.current_model != resolved_model:
                        await self._apply_model_config(session_id, resolved_model, sess_info)

                    logger.debug("Resumed existing ACP session: %s", session_id)
                    return session_id, True
                else:
                    logger.info("Session %s expired and will be recreated", session_id)
                    self._sessions.pop(session_id, None)

            # 2. Need to create a new session
            self._prune_expired()

            session_res = await self.client.session_new(
                cwd=settings.workspace_dir,
                system_prompt=system_prompt,
            )
            new_session_id = session_res["sessionId"]

            # Initialize session model info
            sess_info = ACPSessionInfo(
                session_id=new_session_id,
                created_at=now,
                last_accessed=now,
                current_model=settings.default_model,
                current_thought_level="high",
                turn_count=1,
            )

            # Configure requested model if not default
            if resolved_model != settings.default_model:
                await self._apply_model_config(new_session_id, resolved_model, sess_info)

            self._sessions[new_session_id] = sess_info
            logger.info("Created new ACP session: %s (model: %s)", new_session_id, resolved_model)
            return new_session_id, False

    async def _apply_model_config(
        self, session_id: str, resolved_model: str, sess_info: ACPSessionInfo
    ) -> None:
        """Apply ACP model and thought_level configuration options to a session."""
        acp_model, thought_level = get_acp_model_config(resolved_model)
        try:
            await self.client.session_set_config_option(session_id, "model", acp_model)
            sess_info.current_model = resolved_model

            if thought_level:
                try:
                    await self.client.session_set_config_option(
                        session_id, "thought_level", thought_level
                    )
                    sess_info.current_thought_level = thought_level
                except Exception as e:
                    logger.debug("thought_level config not accepted for model %s: %s", acp_model, e)
        except Exception as e:
            logger.warning("Could not set model '%s' on session %s: %s", acp_model, session_id, e)

    def _prune_expired(self) -> None:
        """Remove sessions that exceeded TTL or if capacity limit is reached."""
        now = time.time()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if (now - s.last_accessed) > settings.session_ttl_seconds
        ]
        for sid in expired:
            self._sessions.pop(sid, None)

        # If still over limit, drop oldest
        if len(self._sessions) >= settings.max_active_sessions:
            sorted_by_accessed = sorted(self._sessions.items(), key=lambda item: item[1].last_accessed)
            excess = len(self._sessions) - settings.max_active_sessions + 1
            for sid, _ in sorted_by_accessed[:excess]:
                self._sessions.pop(sid, None)

    def get_active_session_count(self) -> int:
        return len(self._sessions)
