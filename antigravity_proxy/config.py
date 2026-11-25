"""Configuration settings for Antigravity ACP to OpenAI Proxy."""

import os
import sys
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ANTIGRAVITY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server configuration
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    # Antigravity ACP executable
    agy_bin: str = "agy.exe" if sys.platform == "win32" else "agy"
    agy_extra_args: list[str] = ["--dangerously-skip-permissions"]

    # Optional API key for authenticating incoming requests to this proxy
    api_key: str | None = None

    # Workspace directory passed to session/new
    workspace_dir: str = str(Path.cwd())

    # Default model if none specified
    default_model: str = "gemini-3.8-flash-high"

    # Session management
    session_ttl_seconds: int = 1800  # 30 minutes
    max_active_sessions: int = 100


settings = Settings()
