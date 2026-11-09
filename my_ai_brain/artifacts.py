"""Save and load local model artifacts without deserializing arbitrary objects."""

import json
from pathlib import Path

import torch

from my_ai_brain.model import ModelConfig, TinyTransformerLM
from my_ai_brain.tokenizer import CharacterTokenizer


def save_artifacts(model: TinyTransformerLM, tokenizer: CharacterTokenizer, directory: str | Path) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    (path / "model_config.json").write_text(
        json.dumps(model.config.to_dict(), indent=2), encoding="utf-8"
    )
    tokenizer.save(path / "tokenizer.json")


def load_artifacts(directory: str | Path, device: str = "cpu") -> tuple[TinyTransformerLM, CharacterTokenizer]:
    path = Path(directory)
    config = ModelConfig(**json.loads((path / "model_config.json").read_text(encoding="utf-8")))
    tokenizer = CharacterTokenizer.load(path / "tokenizer.json")
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("Tokenizer vocabulary size does not match model configuration")
    model = TinyTransformerLM(config)
    state_dict = torch.load(path / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    return model, tokenizer