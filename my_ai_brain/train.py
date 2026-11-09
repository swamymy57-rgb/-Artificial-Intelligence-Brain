"""Command-line training entry point."""

import argparse
from pathlib import Path

import torch

from my_ai_brain.artifacts import save_artifacts
from my_ai_brain.model import ModelConfig, TinyTransformerLM
from my_ai_brain.tokenizer import CharacterTokenizer
from my_ai_brain.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the tiny character-level Transformer")
    parser.add_argument("--text-file", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("artifacts"), type=Path)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--steps-per-epoch", default=20, type=int)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()

    text = arguments.text_file.read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer.train(text)
    torch.manual_seed(arguments.seed)
    model = TinyTransformerLM(ModelConfig(vocab_size=tokenizer.vocab_size))
    losses = train_model(
        model,
        tokenizer,
        text,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        steps_per_epoch=arguments.steps_per_epoch,
        device=arguments.device,
    )
    save_artifacts(model, tokenizer, arguments.output_dir)
    print(f"Saved artifacts to {arguments.output_dir}; final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()