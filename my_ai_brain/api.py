"""FastAPI application for locally hosted generation and SQLite memory."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from my_ai_brain.artifacts import load_artifacts
from my_ai_brain.chat import chat
from my_ai_brain.config import Settings
from my_ai_brain.inference import InferenceEngine
from my_ai_brain.local_model import TextGenerator
from my_ai_brain.memory import SQLiteMemory


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=40, ge=0, le=512)


class MemoryRequest(BaseModel):
    content: str = Field(min_length=1, max_length=10000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=40, ge=0, le=512)


def create_app(
    engine: TextGenerator | None = None,
    memory: SQLiteMemory | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    configured = settings or Settings.from_env()
    active_memory = memory or SQLiteMemory(configured.database)
    active_engine = engine
    if active_engine is None and Path(configured.model_dir).is_dir():
        model, tokenizer = load_artifacts(configured.model_dir, configured.device)
        active_engine = InferenceEngine(model, tokenizer, configured.device)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield

    application = FastAPI(title="MY-AI-BRAIN", version="0.1.0", lifespan=lifespan)

    @application.get("/health")
    def health() -> dict[str, str | bool]:
        return {"status": "ok", "model_loaded": active_engine is not None}

    @application.post("/generate")
    def generate(request: GenerateRequest) -> dict[str, str]:
        if active_engine is None:
            raise HTTPException(status_code=503, detail="No model artifacts are loaded")
        return {"text": active_engine.generate(request.prompt, max_new_tokens=request.max_new_tokens)}

    @application.post("/chat")
    def chat_endpoint(request: ChatRequest) -> dict[str, object]:
        if active_engine is None:
            raise HTTPException(status_code=503, detail="No model artifacts are loaded")
        return chat(active_engine, active_memory, request.message, request.max_new_tokens)

    @application.post("/memory", status_code=201)
    def add_memory(request: MemoryRequest) -> dict[str, int]:
        return {"id": active_memory.add(request.content)}

    @application.get("/memory")
    def search_memory(
        query: str = Query(default="", max_length=500), limit: int = Query(default=20, ge=1, le=100)
    ) -> dict[str, list[dict[str, int | str]]]:
        return {"items": active_memory.search(query, limit)}

    return application


app = create_app()