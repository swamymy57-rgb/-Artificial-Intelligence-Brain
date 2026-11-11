
"""Save and load local AI model artifacts safely."""

import json
from pathlib import Path

import torch

from my_ai_brain.model import ModelConfig, TinyTransformerLM
from my_ai_brain.tokenizer import CharacterTokenizer


def save_artifacts(
    model: TinyTransformerLM,
    tokenizer: CharacterTokenizer,
    directory: str | Path,
) -> None:
    path = Path(directory).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path / "model.pt")

    (path / "model_config.json").write_text(
        json.dumps(model.config.to_dict(), indent=2),
        encoding="utf-8",
    )

    tokenizer.save(path / "tokenizer.json")


def load_artifacts(
    directory: str | Path,
    device: str = "cpu",
) -> tuple[TinyTransformerLM, CharacterTokenizer]:

    path = Path(directory).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"AI model directory not found: {path}. "
            "Please train the model first."
        )

    if not path.is_dir():
        raise NotADirectoryError(
            f"Model path is not a directory: {path}"
        )

    required_files = [
        "model.pt",
        "model_config.json",
        "tokenizer.json",
    ]

    missing_files = [
        name for name in required_files
        if not (path / name).is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Missing AI model files in {path}: "
            + ", ".join(missing_files)
            + ". Please train the model first."
        )

    try:
        config_data = json.loads(
            (path / "model_config.json").read_text(
                encoding="utf-8"
            )
        )

        config = ModelConfig(**config_data)

        tokenizer = CharacterTokenizer.load(
            path / "tokenizer.json"
        )

        if tokenizer.vocab_size != config.vocab_size:
            raise ValueError(
                "Tokenizer vocabulary size does not match "
                "model configuration."
            )

        model = TinyTransformerLM(config)

        state_dict = torch.load(
            path / "model.pt",
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        return model, tokenizer

    except Exception as error:
        raise RuntimeError(
            f"Failed to load AI model from {path}: {error}"
        ) from error
        
