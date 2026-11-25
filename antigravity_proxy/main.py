"""CLI entry point for Antigravity ACP to OpenAI Proxy."""

import argparse
import logging
import sys
import uvicorn

from antigravity_proxy.config import settings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Antigravity ACP to OpenAI Compatible Proxy Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default=settings.host,
        help="Host interface to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help="Port to run the HTTP proxy server on",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=settings.api_key,
        help="Optional API key required for requests (Bearer token)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.default_model,
        help="Default model identifier to route requests to",
    )
    parser.add_argument(
        "--agy-bin",
        type=str,
        default=settings.agy_bin,
        help="Path or name of the Antigravity CLI executable",
    )
    parser.add_argument(
        "--workspace-dir",
        type=str,
        default=settings.workspace_dir,
        help="Directory to use as workspace for Antigravity sessions",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=settings.log_level,
        help="Logging level",
    )
    return parser.parse_args()


def print_banner(host: str, port: int, model: str, api_key: str | None):
    print("=" * 65)
    print("  🚀  Antigravity ACP -> OpenAI Compatible Proxy Server")
    print("=" * 65)
    print(f"  • Base URL:          http://{host}:{port}/v1")
    print(f"  • Chat Completions:  http://{host}:{port}/v1/chat/completions")
    print(f"  • Models Endpoint:   http://{host}:{port}/v1/models")
    print(f"  • Health Check:      http://{host}:{port}/health")
    print(f"  • Default Model:     {model}")
    print(f"  • Authentication:    {'Required (Bearer ' + api_key + ')' if api_key else 'None (Open)'}")
    print("=" * 65)
    print("  Ready to accept requests from Cursor, Continue, Aider,")
    print("  OpenAI Python SDK, LangChain, or curl!")
    print("=" * 65)
    print()


def main():
    args = parse_args()

    # Update settings from CLI flags
    settings.host = args.host
    settings.port = args.port
    settings.api_key = args.api_key
    settings.default_model = args.model
    settings.agy_bin = args.agy_bin
    settings.workspace_dir = args.workspace_dir
    settings.log_level = args.log_level

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print_banner(settings.host, settings.port, settings.default_model, settings.api_key)

    uvicorn.run(
        "antigravity_proxy.server:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
