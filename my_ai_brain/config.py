"""Environment-based settings without loading secrets from files implicitly."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    model_dir: str = "artifacts"
    database: str = "data/memory.sqlite3"
    device: str = "cpu"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_dir=os.environ.get("MY_AI_BRAIN_MODEL_DIR", "artifacts"),
            database=os.environ.get("MY_AI_BRAIN_DATABASE", "data/memory.sqlite3"),
            device=os.environ.get("MY_AI_BRAIN_DEVICE", "cpu"),
        )