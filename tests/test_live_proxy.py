"""End-to-end live test verifying OpenAI SDK and HTTP client against the proxy."""

import subprocess
import sys
import time
import httpx
from openai import OpenAI


def run_live_tests():
    port = 8765
    base_url = f"http://127.0.0.1:{port}/v1"

    # Start the proxy server process
    cmd = [
        sys.executable,
        "-m",
        "antigravity_proxy.main",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--log-level",
        "INFO",
    ]

    print(f"Starting proxy on port {port}...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    try:
        # Wait for server to be healthy
        ready = False
        health_url = f"http://127.0.0.1:{port}/health"
        for _ in range(30):
            try:
                r = httpx.get(health_url, timeout=1.0)
                if r.status_code == 200 and r.json().get("status") == "healthy":
                    ready = True
                    break
            except Exception:
                time.sleep(0.5)

        if not ready:
            print("Server failed to start within timeout. Server logs:")
            proc.terminate()
            out, _ = proc.communicate(timeout=5)
            print(out)
            sys.exit(1)

        print("Server is healthy! Running tests with OpenAI Python SDK...")

        client = OpenAI(
            base_url=base_url,
            api_key="any-key-is-accepted-by-default",
        )

        # 1. Test Models Listing
        print("\n--- Test 1: Models Listing ---")
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        print(f"Found {len(model_ids)} models. Sample: {model_ids[:5]}")
        assert "gemini-3.8-flash-high" in model_ids
        assert "gpt-4o" in model_ids
        print("✓ Models listing passed.")

        # 2. Test Non-Streaming Chat Completion
        print("\n--- Test 2: Non-Streaming Chat Completion ---")
        completion = client.chat.completions.create(
            model="gemini-3.8-flash-high",
            messages=[
                {"role": "system", "content": "You are a concise AI assistant."},
                {"role": "user", "content": "What is 7 * 8? Answer with only the number."},
            ],
            stream=False,
        )
        print("Completion response ID:", completion.id)
        reply = completion.choices[0].message.content.strip()
        print("Response Content:", reply)
        assert "56" in reply
        print("✓ Non-streaming chat completion passed.")

        # 3. Test Streaming Chat Completion
        print("\n--- Test 3: Streaming Chat Completion ---")
        stream = client.chat.completions.create(
            model="gemini-3.8-flash-high",
            messages=[
                {"role": "user", "content": "Count from 1 to 3 with spaces in between."},
            ],
            stream=True,
        )
        collected_chunks = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                collected_chunks.append(chunk.choices[0].delta.content)

        streamed_text = "".join(collected_chunks)
        print("Streamed Output:", streamed_text)
        assert len(collected_chunks) > 0
        assert "1" in streamed_text and "2" in streamed_text and "3" in streamed_text
        print("✓ Streaming chat completion passed.")

        # 4. Test Multi-Turn Continuity using Session ID
        print("\n--- Test 4: Multi-Turn Conversation Continuity ---")
        turn1 = client.chat.completions.create(
            model="gemini-3.8-flash-high",
            messages=[
                {"role": "user", "content": "Remember this password: ORANGE-999."},
            ],
            stream=False,
        )
        print("Turn 1 completed.")

        # Using the same conversation with the same history:
        turn2 = client.chat.completions.create(
            model="gemini-3.8-flash-high",
            messages=[
                {"role": "user", "content": "Remember this password: ORANGE-999."},
                {"role": "assistant", "content": turn1.choices[0].message.content},
                {"role": "user", "content": "What was the password I told you?"},
            ],
            stream=False,
        )
        turn2_reply = turn2.choices[0].message.content
        print("Turn 2 Reply:", turn2_reply)
        assert "ORANGE-999" in turn2_reply
        print("✓ Multi-turn conversation continuity passed.")

        print("\n🎉 ALL LIVE TESTS PASSED SUCCESSFULLY! 🎉")

    finally:
        print("\nStopping proxy server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Proxy server stopped.")


if __name__ == "__main__":
    run_live_tests()
