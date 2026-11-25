# Antigravity ACP to OpenAI Proxy

> **High-performance OpenAI-compatible REST & SSE Streaming proxy backed by Google Antigravity Agent Client Protocol (ACP).**

This server acts as a bridge between any standard OpenAI client (Cursor, Continue, Aider, LangChain, LiteLLM, LibreChat, OpenAI SDK, curl) and Google's **Antigravity CLI** (`agy acp`). It translates incoming OpenAI Chat Completions requests into Agent Client Protocol (ACP) JSON-RPC messages and streams responses back in standard OpenAI format with full support for reasoning tokens (`reasoning_content`), multi-turn memory, and multimodal inputs.

---

## 🌟 Key Features

- **Standard OpenAI Endpoints**:
  - `POST /v1/chat/completions` (Streaming SSE & Non-Streaming JSON)
  - `GET /v1/models` and `GET /v1/models/{model}`
  - `GET /health`
- **Native Chain-of-Thought / Reasoning Streaming**:
  - Emits `agent_thought_chunk` as `delta.reasoning_content` (supported by modern clients like Cursor, Continue, and CherryStudio).
- **Zero-Latency Warm Sessions**:
  - Manages a persistent asynchronous `agy acp` process pool. Session spin-up takes less than 20ms.
- **Stateful & Stateless Multi-Turn Support**:
  - Works with standard stateless OpenAI clients sending full message histories.
  - Supports sticky sessions via `X-Session-ID` header or `session_id` in request payloads for maximum token efficiency and instant multi-turn interactions.
- **Dynamic Model Switching & Intelligent Aliasing**:
  - Seamlessly route between `gemini-3.8-flash-high`, `gemini-3.8-flash-medium`, `gemini-3.8-flash-low`, `claude-sonnet-4-6`, `claude-opus-4-6-thinking`, and `gpt-oss-120b-medium`.
  - Common aliases (`gpt-4o`, `claude-3-7-sonnet`, `gemini-flash`, etc.) are automatically mapped to the optimal Antigravity model.
- **Heavy In-Memory Model Cache**:
  - Models are inspected once on server boot from Antigravity ACP and frozen in memory. `GET /v1/models` and `GET /v1/models/{model}` return in < 1ms with zero live ACP queries.
- **Multimodal Image Support**:
  - Handles `data:image/...;base64,...` data URIs and HTTP image URLs, converting them into native ACP image blocks.
- **Production Ready**:
  - Full CORS support for browser frontends, configurable Bearer token authentication, cooperative cancellation, and automated session pruning.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────┐
│ OpenAI-Compatible Client                               │
│ (Cursor / Continue / Aider / OpenAI SDK / curl / etc.) │
└──────────────────────────┬─────────────────────────────┘
                           │ HTTP REST / SSE Stream
                           ▼
┌────────────────────────────────────────────────────────┐
│ Antigravity ACP Proxy (FastAPI + Uvicorn)              │
│ ┌──────────────────────┐    ┌────────────────────────┐ │
│ │ Model / Alias Router │    │ Session & LRU Manager  │ │
│ └──────────────────────┘    └────────────────────────┘ │
│ ┌────────────────────────────────────────────────────┐ │
│ │ Async JSON-RPC Engine (acp_client.py)              │ │
│ └────────────────────────────────────────────────────┘ │
└──────────────────────────┬─────────────────────────────┘
                           │ stdio (JSON-RPC 2.0 lines)
                           ▼
┌────────────────────────────────────────────────────────┐
│ Google Antigravity Agent Client Protocol Server        │
│ (`agy.exe acp --dangerously-skip-permissions`)         │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart

### 1. Requirements

- Python 3.10+
- Google Antigravity CLI (`agy`) installed and logged in (`agy --help`)

### 2. Install Dependencies

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install fastapi uvicorn pydantic pydantic-settings httpx
```

### 3. Start the Server

```bash
python -m antigravity_proxy.main --port 8000 --host 127.0.0.1
```

Or using the installed console script:

```bash
antigravity-acp-proxy --port 8000
```

---

## 💻 Usage Examples

### Python (Official OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed",  # or your configured ANTIGRAVITY_API_KEY
)

# 1. Non-streaming completion
response = client.chat.completions.create(
    model="gemini-3.8-flash-high",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a python function for quicksort."},
    ],
    stream=False,
)
print(response.choices[0].message.content)

# 2. Streaming completion with reasoning
stream = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[
        {"role": "user", "content": "Explain how RSA encryption works in 3 paragraphs."},
    ],
    stream=True,
)

for chunk in stream:
    # If the model emitted reasoning tokens
    if hasattr(chunk.choices[0].delta, "reasoning_content") and chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
    # Output content tokens
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

### cURL

#### Non-Streaming:
```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.8-flash-high",
    "messages": [
      {"role": "user", "content": "Hello! What is the speed of light?"}
    ]
  }'
```

#### Streaming:
```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [
      {"role": "user", "content": "Count from 1 to 5."}
    ],
    "stream": true
  }'
```

#### List Models:
```bash
curl http://127.0.0.1:8000/v1/models
```

---

## 🛠️ Editor & Tool Integration

### Cursor IDE
1. Go to **Cursor Settings** -> **Models**.
2. Under **OpenAI API Key**, enter any placeholder key (e.g. `antigravity`).
3. Click **Override OpenAI Base URL** and set:
   ```
   http://127.0.0.1:8000/v1
   ```
4. Add models: `gemini-3.8-flash-high`, `claude-sonnet-4-6`, `claude-opus-4-6-thinking`.

### Continue (VS Code & JetBrains)
In your `~/.continue/config.json`:
```json
{
  "models": [
    {
      "title": "Antigravity Gemini Flash",
      "provider": "openai",
      "model": "gemini-3.8-flash-high",
      "apiBase": "http://127.0.0.1:8000/v1",
      "apiKey": "none"
    },
    {
      "title": "Antigravity Claude Sonnet",
      "provider": "openai",
      "model": "claude-sonnet-4-6",
      "apiBase": "http://127.0.0.1:8000/v1",
      "apiKey": "none"
    }
  ]
}
```

### LibreChat / Chatbox / NextChat
- **API URL**: `http://127.0.0.1:8000/v1`
- **API Key**: Any value (unless `--api-key` is configured)

---

## ⚙️ Configuration & Environment Variables

All settings can be configured via CLI flags or environment variables prefixed with `ANTIGRAVITY_`:

| Flag | Env Variable | Default | Description |
|---|---|---|---|
| `--host` | `ANTIGRAVITY_HOST` | `127.0.0.1` | Network interface to bind |
| `--port` | `ANTIGRAVITY_PORT` | `8000` | Port to listen on |
| `--api-key` | `ANTIGRAVITY_API_KEY` | `None` | Optional Bearer auth token required from clients |
| `--model` | `ANTIGRAVITY_DEFAULT_MODEL` | `gemini-3.8-flash-high` | Default model when request omits `model` |
| `--agy-bin` | `ANTIGRAVITY_AGY_BIN` | `agy.exe` / `agy` | Path to Antigravity CLI binary |
| `--workspace-dir` | `ANTIGRAVITY_WORKSPACE_DIR` | Current directory | Working directory for ACP sessions |
| `--log-level` | `ANTIGRAVITY_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## 🧪 Running Tests

Run the full pytest suite:

```bash
pytest -v
```

Run live integration tests against Antigravity ACP:

```bash
python tests/test_live_proxy.py
```

---

## 📄 License

MIT License.
