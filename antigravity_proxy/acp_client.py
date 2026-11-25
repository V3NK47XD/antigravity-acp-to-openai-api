"""Agent Client Protocol (ACP) Async JSON-RPC Client."""

import asyncio
import json
import logging
import sys
from typing import Any, AsyncGenerator

logger = logging.getLogger("antigravity_proxy.acp")


class ACPError(Exception):
    """Exception raised when an ACP request returns an error."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(f"ACP Error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


class ACPClient:
    """Async client managing communication with an 'agy acp' subprocess."""

    def __init__(
        self,
        executable: str = "agy.exe" if sys.platform == "win32" else "agy",
        extra_args: list[str] | None = None,
        workspace_dir: str | None = None,
    ):
        self.executable = executable
        self.extra_args = extra_args or ["--dangerously-skip-permissions"]
        self.workspace_dir = workspace_dir

        self._process: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self._lock = asyncio.Lock()
        self._pending_requests: dict[int, asyncio.Future[dict]] = {}
        self._session_queues: dict[str, asyncio.Queue[dict | None]] = {}

        self._read_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._is_running = False

        self.server_info: dict = {}
        self.agent_capabilities: dict = {}
        self.available_models: list[dict] = []

    async def start(self) -> None:
        """Launch the agy acp subprocess and perform protocol initialization."""
        async with self._lock:
            if self._is_running:
                return

            cmd = [self.executable, "acp"] + self.extra_args
            logger.info("Spawning ACP process: %s", " ".join(cmd))

            try:
                self._process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except Exception as e:
                logger.error("Failed to spawn ACP process '%s': %s", self.executable, e)
                raise RuntimeError(
                    f"Could not start Antigravity ACP process with command: {' '.join(cmd)}. "
                    f"Ensure '{self.executable}' is installed and in your PATH. Error: {e}"
                ) from e

            self._is_running = True
            self._read_task = asyncio.create_task(self._read_loop())
            self._stderr_task = asyncio.create_task(self._stderr_loop())

        # Perform initialize handshake
        await self._initialize()
        logger.info(
            "ACP connection established. Server: %s, Protocol v%s",
            self.server_info.get("name", "Unknown"),
            self.server_info.get("protocolVersion", 1),
        )

    async def _send_raw(self, payload: dict) -> None:
        """Serialize and write a JSON-RPC message line to stdin."""
        if not self._process or not self._process.stdin:
            raise RuntimeError("ACP process is not running")

        line = json.dumps(payload) + "\n"
        self._process.stdin.write(line.encode("utf-8"))
        await self._process.stdin.drain()

    async def send_request(
        self, method: str, params: dict | None = None, timeout: float = 120.0
    ) -> dict:
        """Send a JSON-RPC request and await its response."""
        if not self._is_running:
            await self.start()

        async with self._lock:
            req_id = self._next_id
            self._next_id += 1

            loop = asyncio.get_running_loop()
            future: asyncio.Future[dict] = loop.create_future()
            self._pending_requests[req_id] = future

        msg = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {},
        }

        await self._send_raw(msg)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_requests.pop(req_id, None)
            raise TimeoutError(f"ACP request {method} (id={req_id}) timed out after {timeout}s")
        except Exception:
            self._pending_requests.pop(req_id, None)
            raise

    async def send_notification(self, method: str, params: dict | None = None) -> None:
        """Send a one-way JSON-RPC notification."""
        if not self._is_running:
            await self.start()

        msg = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }
        await self._send_raw(msg)

    async def _read_loop(self) -> None:
        """Continuously read newline-delimited JSON-RPC messages from stdout."""
        assert self._process and self._process.stdout
        reader = self._process.stdout

        while self._is_running:
            try:
                line_bytes = await reader.readline()
                if not line_bytes:
                    logger.warning("ACP stdout reached EOF")
                    break

                line = line_bytes.decode("utf-8").strip()
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError as err:
                    logger.warning("Failed to decode ACP message: %s (raw: %s)", err, line)
                    continue

                await self._handle_message(msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error reading ACP stdout: %s", e)
                break

        self._is_running = False

    async def _stderr_loop(self) -> None:
        """Read and log messages from stderr."""
        assert self._process and self._process.stderr
        reader = self._process.stderr

        while self._is_running:
            try:
                line_bytes = await reader.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8").strip()
                if line:
                    logger.debug("[ACP stderr] %s", line)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Error reading ACP stderr: %s", e)
                break

    async def _handle_message(self, msg: dict) -> None:
        """Route incoming JSON-RPC message: response, request, or notification."""
        req_id = msg.get("id")
        method = msg.get("method")

        # 1. Incoming Request from ACP Agent to Client (has both id and method)
        if req_id is not None and method:
            await self._handle_agent_request(req_id, method, msg.get("params", {}))
            return

        # 2. Response to a pending client request (has id, and result or error)
        if req_id is not None:
            future = self._pending_requests.pop(req_id, None)
            if future and not future.done():
                if "error" in msg:
                    err = msg["error"]
                    future.set_exception(
                        ACPError(err.get("code", -1), err.get("message", "Unknown error"), err.get("data"))
                    )
                else:
                    future.set_result(msg.get("result", {}))
            return

        # 3. Notification from ACP Agent (method without id)
        if method == "session/update":
            params = msg.get("params", {})
            session_id = params.get("sessionId")
            if session_id and session_id in self._session_queues:
                queue = self._session_queues[session_id]
                await queue.put(params.get("update", {}))
            else:
                logger.debug("Received session/update for untracked session %s", session_id)
        else:
            logger.debug("Received unhandled ACP notification: %s", method)

    async def _handle_agent_request(self, req_id: int | str, method: str, params: dict) -> None:
        """Handle incoming requests from ACP agent (e.g. permission or fs requests)."""
        logger.info("Received request from ACP agent: %s (id=%s)", method, req_id)
        # By default auto-approve permission requests or return safe default
        response: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id}

        if "permission" in method.lower():
            # Grant permission
            response["result"] = {"outcome": {"outcome": "approved"}}
        elif method == "fs/readTextFile":
            response["result"] = {"content": ""}
        elif method == "fs/writeTextFile":
            response["result"] = {}
        else:
            response["result"] = {}

        await self._send_raw(response)

    async def _initialize(self) -> dict:
        """Initialize connection with the ACP server."""
        payload = {
            "protocolVersion": 1,
            "clientCapabilities": {
                "fs": {"readTextFile": False, "writeTextFile": False},
                "terminal": False,
            },
            "clientInfo": {
                "name": "antigravity-acp-openai-proxy",
                "version": "1.0.0",
            },
        }
        res = await self.send_request("initialize", payload)
        self.server_info = res.get("agentInfo", {})
        self.server_info["protocolVersion"] = res.get("protocolVersion", 1)
        self.agent_capabilities = res.get("agentCapabilities", {})
        return res

    async def session_new(
        self,
        cwd: str | None = None,
        system_prompt: str | list[dict] | None = None,
        mcp_servers: list | None = None,
    ) -> dict:
        """Create a new ACP session."""
        params: dict[str, Any] = {
            "cwd": cwd or self.workspace_dir or ".",
            "mcpServers": mcp_servers or [],
        }
        if system_prompt:
            if isinstance(system_prompt, str):
                params["systemPrompt"] = [{"type": "text", "text": system_prompt}]
            elif isinstance(system_prompt, list):
                params["systemPrompt"] = system_prompt

        res = await self.send_request("session/new", params)
        if "models" in res and "availableModels" in res["models"]:
            self.available_models = res["models"]["availableModels"]
        return res

    async def session_set_config_option(
        self, session_id: str, config_id: str, value: Any
    ) -> dict:
        """Set a configuration option (such as model, thought_level, or mode) on an active session."""
        params = {
            "sessionId": session_id,
            "configId": config_id,
            "value": value,
        }
        return await self.send_request("session/set_config_option", params)

    async def session_prompt_stream(
        self, session_id: str, prompt_blocks: list[dict]
    ) -> AsyncGenerator[dict, None]:
        """Stream session/update events for a prompt turn until the prompt completes."""
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        self._session_queues[session_id] = queue

        req_id = self._next_id
        self._next_id += 1

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict] = loop.create_future()
        self._pending_requests[req_id] = future

        msg = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": prompt_blocks,
            },
        }

        await self._send_raw(msg)

        try:
            # Yield updates from the queue until the request future completes
            while True:
                # Check if future has completed or error
                if future.done():
                    # Drain any remaining updates in queue
                    while not queue.empty():
                        update = queue.get_nowait()
                        if update is not None:
                            yield {"type": "update", "data": update}

                    result = future.result()
                    yield {"type": "final", "data": result}
                    break

                try:
                    # Wait for next update or future completion
                    update = await asyncio.wait_for(queue.get(), timeout=0.1)
                    if update is not None:
                        yield {"type": "update", "data": update}
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.info("Prompt stream cancelled for session %s, sending cancel notification", session_id)
            await self.cancel_prompt(session_id)
            raise
        finally:
            self._session_queues.pop(session_id, None)
            self._pending_requests.pop(req_id, None)

    async def cancel_prompt(self, session_id: str) -> None:
        """Send session/cancel notification to abort active prompt turn."""
        await self.send_notification("session/cancel", {"sessionId": session_id})

    async def close(self) -> None:
        """Terminate the ACP subprocess and cleanup background tasks."""
        self._is_running = False

        if self._read_task and not self._read_task.done():
            self._read_task.cancel()
        if self._stderr_task and not self._stderr_task.done():
            self._stderr_task.cancel()

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=3.0)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

        # Cancel any pending requests
        for fut in self._pending_requests.values():
            if not fut.done():
                fut.cancel()
        self._pending_requests.clear()
        self._session_queues.clear()
        logger.info("ACP client closed")
