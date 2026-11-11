"""Environment-based settings for MY AI BRAIN."""

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    model_dir: str = "artifacts"
    database: str = "data/memory.sqlite3"
    device: str = "cpu"

    @classmethod
    def from_env(cls) -> "Settings":
        # Project root directory
        project_root = Path(__file__).resolve().parent.parent

        # Model directory
        model_dir = Path(
            os.environ.get(
                "MY_AI_BRAIN_MODEL_DIR",
                str(project_root / "artifacts")
            )
        )

        # Database path
        database = Path(
            os.environ.get(
                "MY_AI_BRAIN_DATABASE",
                str(project_root / "data" / "memory.sqlite3")
            )
        )

        # Device
        device = os.environ.get(
            "MY_AI_BRAIN_DEVICE", "cpu"
        )

        return cls(
            model_dir=str(model_dir),
            database=str(database),
            device=device,
        )
        
