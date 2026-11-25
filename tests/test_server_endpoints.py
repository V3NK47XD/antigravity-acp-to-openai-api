"""Unit tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from antigravity_proxy.server import app
from antigravity_proxy.config import settings


@pytest.fixture
def client():
    # Use TestClient without lifespan for fast unit testing of pure routes
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "antigravity-acp-openai-proxy"


def test_list_models_endpoint(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0

    model_ids = [m["id"] for m in data["data"]]
    assert "gemini-3.8-flash-high" in model_ids
    assert "claude-sonnet-4-6" in model_ids
    assert "gpt-4o" in model_ids


def test_get_single_model(client):
    response = client.get("/v1/models/gemini-3.8-flash-high")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "gemini-3.8-flash-high"
    assert data["object"] == "model"


def test_heavy_cached_models_in_memory(client):
    from antigravity_proxy import server

    # Modify the in-memory cache directly to prove the endpoint reads strictly from memory
    custom_model_id = "test-in-memory-cached-model"
    server.cached_models_dict[custom_model_id] = server.ModelObject(
        id=custom_model_id,
        created=1234567890,
        owned_by="antigravity",
    )
    server.cached_models_response.data.append(server.cached_models_dict[custom_model_id])

    # Check that GET /v1/models reflects the memory cache instantly
    res_list = client.get("/v1/models")
    assert res_list.status_code == 200
    ids = [m["id"] for m in res_list.json()["data"]]
    assert custom_model_id in ids

    # Check that GET /v1/models/{model_id} serves directly from the memory dict
    res_single = client.get(f"/v1/models/{custom_model_id}")
    assert res_single.status_code == 200
    assert res_single.json()["id"] == custom_model_id

    # Check non-existent model returns 404
    res_404 = client.get("/v1/models/non-existent-model-xyz")
    assert res_404.status_code == 404


def test_auth_rejection_when_key_set(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret-test-key")

    # Missing auth header
    res_no_auth = client.get("/v1/models")
    assert res_no_auth.status_code == 401

    # Wrong auth header
    res_wrong_auth = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
    assert res_wrong_auth.status_code == 401

    # Correct auth header
    res_ok = client.get("/v1/models", headers={"Authorization": "Bearer secret-test-key"})
    assert res_ok.status_code == 200
