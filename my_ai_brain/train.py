
"""Reliable training entry point for MY AI BRAIN."""

import argparse
import json
import random
from pathlib import Path

import torch

from my_ai_brain.artifacts import save_artifacts
from my_ai_brain.model import ModelConfig, TinyTransformerLM
from my_ai_brain.tokenizer import CharacterTokenizer
from my_ai_brain.training import evaluate_model, train_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Train MY AI BRAIN"
    )

    parser.add_argument(
        "--text-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "sample_telugu_english.txt"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts"
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    if not 0 < args.eval_fraction < 1:
        parser.error("eval-fraction must be between 0 and 1")

    text_file = args.text_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not text_file.is_file():
        parser.error(f"Training dataset not found: {text_file}")

    text = text_file.read_text(encoding="utf-8")

    if len(text.strip()) < 4:
        parser.error("Training dataset is too small")

    output_dir.mkdir(parents=True, exist_ok=True)

    records = [
        line for line in text.splitlines()
        if line.strip()
    ]

    if len(records) >= 2:
        order = list(range(len(records)))
        random.Random(args.seed).shuffle(order)

        eval_count = min(
            len(records) - 1,
            max(1, round(len(records) * args.eval_fraction))
        )

        eval_indices = set(order[:eval_count])

        training_text = "\n".join(
            record for i, record in enumerate(records)
            if i not in eval_indices
        )

        evaluation_text = "\n".join(
            record for i, record in enumerate(records)
            if i in eval_indices
        )
    else:
        split_at = int(len(text) * (1 - args.eval_fraction))

        if split_at < 2 or len(text) - split_at < 2:
            parser.error("Dataset must contain at least four characters")

        training_text = text[:split_at]
        evaluation_text = text[split_at:]

    tokenizer = CharacterTokenizer.train(text)

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    model = TinyTransformerLM(
        ModelConfig(vocab_size=tokenizer.vocab_size)
    )

    print("Starting AI model training...")

    losses = train_model(
        model,
        tokenizer,
        training_text,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_dir=output_dir / "checkpoints",
    )

    save_artifacts(model, tokenizer, output_dir)

    metrics = {
        "train_loss": losses[-1],
        "validation": evaluate_model(
            model,
            tokenizer,
            evaluation_text,
            device=args.device,
        ),
    }

    training_config = {
        "text_file": str(text_file),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "steps_per_epoch": args.steps_per_epoch,
        "learning_rate": args.learning_rate,
        "eval_fraction": args.eval_fraction,
        "seed": args.seed,
        "device": args.device,
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    (output_dir / "training_config.json").write_text(
        json.dumps(training_config, indent=2) + "\n",
        encoding="utf-8",
    )

    print("AI MODEL TRAINING COMPLETED")
    print(f"Model saved at: {output_dir}")
    print(f"Train loss: {metrics['train_loss']:.4f}")


if __name__ == "__main__":
    main()
    
