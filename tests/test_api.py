import pytest
from fastapi.testclient import TestClient

from polyrag import api
from polyrag.config import settings
from polyrag.index.chunker import chunk_records
from polyrag.index.embedder import HashingEmbedder
from polyrag.index.store import VectorStore
from tests.conftest import RECORDS


@pytest.fixture()
def client(tmp_path, monkeypatch):
    embedder = HashingEmbedder(dim=256)
    chunks = chunk_records(RECORDS, max_chars=400, overlap=50)
    VectorStore.build(chunks, embedder).save(tmp_path)
    monkeypatch.setattr(settings, "index_dir", tmp_path)
    monkeypatch.setattr(settings, "embedder", "hashing")
    monkeypatch.setattr(settings, "groq_api_key", "test-groq-key")
    monkeypatch.setattr(settings, "github_token", "")
    monkeypatch.setattr(settings, "openrouter_api_key", "")
    monkeypatch.setattr(settings, "ollama_enabled", False)
    with TestClient(api.app) as c:
        yield c


def test_healthz_reports_index(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["index"]["num_chunks"] > 0


def test_models_endpoint(client):
    models = {m["alias"]: m for m in client.get("/models").json()}
    assert {"llama", "gpt-oss", "gpt-oss-20b", "qwen-3.6", "deepseek", "mock"} <= models.keys()
    assert models["gpt-oss"]["selectable"] is True
    assert models["qwen-3.6"]["selectable"] is True
    assert models["deepseek"]["selectable"] is False
    assert models["llama"]["archived"] is True
    assert models["llama"]["ready"] is False
    assert models["llama"]["selectable"] is False
    assert models["llama"]["replacement"] == "gpt-oss"


def test_ask_with_mock(client):
    resp = client.post("/ask", json={"question": "risk functions?", "model": "mock", "k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["citation_coverage"] == 1.0
    assert len(body["sources"]) == 2


def test_compare_validates_aliases(client):
    resp = client.post("/compare", json={"question": "q?", "models": ["nope"]})
    assert resp.status_code == 422


def test_archived_alias_returns_replacement_without_provider_call(client):
    resp = client.post("/ask", json={"question": "risk functions?", "model": "llama"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == ""
    assert "archived" in body["model"]["error"]
    assert "gpt-oss" in body["model"]["error"]
