"""Minimal next-character training loop for the tiny Transformer."""

import torch

from my_ai_brain.model import TinyTransformerLM
from my_ai_brain.tokenizer import CharacterTokenizer


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
    model.eval()
    return losses