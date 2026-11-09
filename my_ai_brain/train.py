"""Command-line training entry point."""

import argparse
import json
import random
from pathlib import Path

import torch

from my_ai_brain.artifacts import save_artifacts
from my_ai_brain.model import ModelConfig, TinyTransformerLM
from my_ai_brain.tokenizer import CharacterTokenizer
from my_ai_brain.training import evaluate_model, train_model


def main() -> None:
    bundled_corpus = Path(__file__).resolve().parent.parent / "data" / "sample_telugu_english.txt"
    parser = argparse.ArgumentParser(description="Train and evaluate the tiny character-level Transformer")
    parser.add_argument("--text-file", default=bundled_corpus, type=Path)
    parser.add_argument("--output-dir", default=Path("artifacts"), type=Path)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--steps-per-epoch", default=10, type=int)
    parser.add_argument("--learning-rate", default=3e-4, type=float)
    parser.add_argument("--eval-fraction", default=0.2, type=float)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()

    if not 0 < arguments.eval_fraction < 1:
        parser.error("--eval-fraction must be between 0 and 1")
    text = arguments.text_file.read_text(encoding="utf-8")
    records = [line for line in text.splitlines() if line.strip()]
    if len(records) >= 2:
        order = list(range(len(records)))
        random.Random(arguments.seed).shuffle(order)
        eval_count = min(len(records) - 1, max(1, round(len(records) * arguments.eval_fraction)))
        eval_indices = set(order[:eval_count])
        training_text = "\n".join(record for i, record in enumerate(records) if i not in eval_indices)
        evaluation_text = "\n".join(record for i, record in enumerate(records) if i in eval_indices)
    else:
        split_at = int(len(text) * (1 - arguments.eval_fraction))
        if split_at < 2 or len(text) - split_at < 2:
            parser.error("Corpus must have at least four characters or two non-empty lines")
        training_text, evaluation_text = text[:split_at], text[split_at:]

    tokenizer = CharacterTokenizer.train(text)
    torch.manual_seed(arguments.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = TinyTransformerLM(ModelConfig(vocab_size=tokenizer.vocab_size))
    losses = train_model(
        model,
        tokenizer,
        training_text,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        steps_per_epoch=arguments.steps_per_epoch,
        learning_rate=arguments.learning_rate,
        device=arguments.device,
        checkpoint_dir=arguments.output_dir / "checkpoints",
    )
    save_artifacts(model, tokenizer, arguments.output_dir)
    metrics = {
        "train_loss": losses[-1],
        "validation": evaluate_model(model, tokenizer, evaluation_text, device=arguments.device),
    }
    training_config = {
        "text_file": str(arguments.text_file),
        "epochs": arguments.epochs,
        "batch_size": arguments.batch_size,
        "steps_per_epoch": arguments.steps_per_epoch,
        "learning_rate": arguments.learning_rate,
        "eval_fraction": arguments.eval_fraction,
        "seed": arguments.seed,
        "device": arguments.device,
    }
    (arguments.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (arguments.output_dir / "training_config.json").write_text(
        json.dumps(training_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved model and checkpoints to {arguments.output_dir}")
    print(f"Train loss: {metrics['train_loss']:.4f}")
    print(
        "Validation: "
        f"loss={metrics['validation']['loss']:.4f}, "
        f"perplexity={metrics['validation']['perplexity']:.4f}, "
        f"accuracy={metrics['validation']['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()