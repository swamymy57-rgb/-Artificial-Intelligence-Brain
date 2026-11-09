"""Run a tiny train, CLI inference, and API memory-retrieval smoke test."""

from pathlib import Path
import subprocess
import sys
import tempfile

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from my_ai_brain.api import create_app
from my_ai_brain.config import Settings


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="my-ai-brain-smoke-") as directory:
        workspace = Path(directory)
        model_dir = workspace / "artifacts"
        database = workspace / "memory.sqlite3"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "my_ai_brain.train",
                "--output-dir",
                str(model_dir),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--steps-per-epoch",
                "1",
                "--seed",
                "7",
            ],
            cwd=ROOT,
            check=True,
        )

        inference = subprocess.run(
            [
                sys.executable,
                "-m",
                "my_ai_brain.chat",
                "--model-dir",
                str(model_dir),
                "--database",
                str(database),
                "--max-new-tokens",
                "2",
                "--message",
                "Hello",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "AI:" in inference.stdout

        app = create_app(settings=Settings(model_dir=str(model_dir), database=str(database)))
        with TestClient(app) as client:
            assert client.get("/health").json()["model_loaded"] is True
            client.post("/chat", json={"message": "Please remember my favorite color is blue."}).raise_for_status()
            response = client.post("/chat", json={"message": "What is my favorite color?"})
            response.raise_for_status()
            assert any("favorite color is blue" in item["content"] for item in response.json()["memories"])
        print("Smoke test passed: training, CLI inference, API health, and persistent chat memory.")


if __name__ == "__main__":
    main()
