"""FastAPI service: /ask, /compare, /models, /healthz + a minimal comparison UI.

The store and embedder load once at startup (they're read-only afterwards);
LLM calls happen per-request. Designed to run identically via uvicorn locally,
in Docker, and on Cloud Run.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from polyrag import rag
from polyrag.config import PROJECT_ROOT, settings
from polyrag.index.embedder import get_embedder
from polyrag.index.store import VectorStore
from polyrag.llm.providers import REGISTRY, available_aliases

STATIC_DIR = PROJECT_ROOT / "static"

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["store"] = VectorStore.load(settings.index_dir)
    manifest = state["store"].manifest
    # Query embeddings must come from the same embedder that built the index.
    state["embedder"] = get_embedder(
        manifest["embedder"], manifest.get("model") or settings.embed_model,
        settings.embed_device, dim=manifest.get("dim"))
    yield
    state.clear()


app = FastAPI(title="polyrag", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    model: str = "llama"
    k: int | None = Field(default=None, ge=1, le=20)


class CompareRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    models: list[str] = Field(default=["llama", "gpt-oss"], min_length=1, max_length=6)
    k: int | None = Field(default=None, ge=1, le=20)


def _check_models(aliases: list[str]) -> None:
    unknown = [a for a in aliases if a not in REGISTRY]
    if unknown:
        raise HTTPException(422, f"Unknown model alias(es): {unknown}. See /models.")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "index": state["store"].manifest}


@app.get("/models")
def models() -> list[dict]:
    return available_aliases()


@app.post("/ask")
async def ask(req: AskRequest) -> dict:
    _check_models([req.model])
    result = await run_in_threadpool(
        rag.ask, req.question, req.model, state["store"], state["embedder"], req.k)
    return result.to_dict()


@app.post("/compare")
async def compare(req: CompareRequest) -> dict:
    _check_models(req.models)
    results = await run_in_threadpool(
        rag.compare, req.question, req.models, state["store"], state["embedder"], req.k)
    return {"question": req.question, "results": [r.to_dict() for r in results]}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
