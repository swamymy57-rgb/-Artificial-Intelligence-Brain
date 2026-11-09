"""Minimal next-character training loop for the tiny Transformer."""

import math
from pathlib import Path

import torch
from torch.nn import functional as F

from my_ai_brain.model import TinyTransformerLM
from my_ai_brain.tokenizer import CharacterTokenizer


def evaluate_model(
    model: TinyTransformerLM, tokenizer: CharacterTokenizer, text: str, *, device: str = "cpu"
) -> dict[str, float]:
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)
    if token_ids.numel() < 2:
        raise ValueError("Evaluation text must contain at least two characters")
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("Tokenizer vocabulary size does not match model configuration")

    model.to(device)
    model.eval()
    context = model.config.max_seq_len
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    with torch.inference_mode():
        for start in range(0, token_ids.numel() - 1, context):
            end = min(start + context, token_ids.numel() - 1)
            inputs = token_ids[start:end]
            targets = token_ids[start + 1 : end + 1]
            logits, _ = model(inputs.unsqueeze(0))
            total_loss += float(
                F.cross_entropy(logits.squeeze(0), targets, reduction="sum")
            )
            correct_tokens += int((logits.squeeze(0).argmax(dim=-1) == targets).sum())
            total_tokens += targets.numel()

    loss = total_loss / total_tokens
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 100.0)),
        "accuracy": correct_tokens / total_tokens,
    }


def train_model(
    model: TinyTransformerLM,
    tokenizer: CharacterTokenizer,
    text: str,
    *,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    steps_per_epoch: int = 20,
    device: str = "cpu",
    checkpoint_dir: str | Path | None = None,
) -> list[float]:
    if epochs < 1 or batch_size < 1 or steps_per_epoch < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, steps_per_epoch, and learning_rate must be positive")
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)
    if token_ids.numel() < 2:
        raise ValueError("Training text must contain at least two characters")
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("Tokenizer vocabulary size does not match model configuration")

    context = min(model.config.max_seq_len, token_ids.numel() - 1)
    if context < 1:
        raise ValueError("Training text is too short")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []
    for _ in range(epochs):
        epoch_losses = []
        for _ in range(steps_per_epoch):
            starts = torch.randint(token_ids.numel() - context, (batch_size,), device=device)
            inputs = torch.stack([token_ids[start : start + context] for start in starts])
            targets = torch.stack([token_ids[start + 1 : start + context + 1] for start in starts])
            _, loss = model(inputs, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(sum(epoch_losses) / len(epoch_losses))
        if checkpoint_dir is not None:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path / f"epoch-{len(losses):03d}.pt")
    model.eval()
    return losses