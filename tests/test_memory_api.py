import stat

from fastapi.testclient import TestClient

from my_ai_brain.artifacts import save_artifacts
from my_ai_brain.api import create_app
from my_ai_brain.config import Settings
from my_ai_brain.memory import SQLiteMemory
from my_ai_brain.model import ModelConfig, TinyTransformerLM
from my_ai_brain.tokenizer import CharacterTokenizer


class StubGenerator:
    def generate(self, prompt: str, *, max_new_tokens: int = 40, **options: object) -> str:
        return prompt + " response"


def test_sqlite_memory_persists_and_searches(tmp_path):
    database = tmp_path / "memory.sqlite3"
    memory = SQLiteMemory(database)
    memory.add("remember this note")
    assert memory.search("note")[0]["content"] == "remember this note"
    assert SQLiteMemory(database).search()[0]["content"] == "remember this note"
    assert stat.S_IMODE(database.stat().st_mode) == 0o600


def test_sqlite_in_memory_database_survives_connections():
    memory = SQLiteMemory(":memory:")
    memory.add("kept in process")
    assert memory.search()[0]["content"] == "kept in process"


def test_api_health_generation_and_memory(tmp_path):
    app = create_app(engine=StubGenerator(), memory=SQLiteMemory(tmp_path / "memory.sqlite3"))
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "model_loaded": True}
        response = client.post("/generate", json={"prompt": "hello"})
        assert response.status_code == 200
        assert response.json()["text"] == "hello response"
        saved = client.post("/memory", json={"content": "a local memory"})
        assert saved.status_code == 201
        assert client.get("/memory", params={"query": "local"}).json()["items"][0]["content"] == "a local memory"


def test_api_chat_retrieves_memory_across_requests(tmp_path):
    app = create_app(engine=StubGenerator(), memory=SQLiteMemory(tmp_path / "memory.sqlite3"))
    with TestClient(app) as client:
        first = client.post("/chat", json={"message": "Please remember my favorite color is blue."})
        assert first.status_code == 200
        assert first.json()["memories"] == []

        second = client.post("/chat", json={"message": "What is my favorite color?"})
        assert second.status_code == 200
        assert any(
            "favorite color is blue" in item["content"]
            for item in second.json()["memories"]
        )
        assert second.json()["reply"] == " response"


def test_api_requires_model_for_generation(tmp_path):
    settings = Settings(model_dir=str(tmp_path / "missing-artifacts"), database=str(tmp_path / "memory.sqlite3"))
    app = create_app(settings=settings)
    with TestClient(app) as client:
        response = client.post("/generate", json={"prompt": "hello"})
        assert response.status_code == 503
        assert client.post("/chat", json={"message": "hello"}).status_code == 503


def test_api_loads_artifacts_and_generates(tmp_path):
    model_dir = tmp_path / "artifacts"
    tokenizer = CharacterTokenizer.train("hello")
    model = TinyTransformerLM(ModelConfig(vocab_size=tokenizer.vocab_size, d_model=8, n_heads=2, n_layers=1))
    save_artifacts(model, tokenizer, model_dir)
    settings = Settings(model_dir=str(model_dir), database=str(tmp_path / "memory.sqlite3"))
    app = create_app(settings=settings)
    with TestClient(app) as client:
        response = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 1})
        assert response.status_code == 200
        assert response.json()["text"].startswith("hi")