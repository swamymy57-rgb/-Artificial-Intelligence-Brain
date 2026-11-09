"""A compact causal Transformer language model implemented with PyTorch."""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.vocab_size < 2 or self.d_model < 1 or self.n_heads < 1 or self.n_layers < 1:
            raise ValueError("Model dimensions must be positive and vocabulary must include at least two tokens")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.max_seq_len < 1 or not 0 <= self.dropout < 1:
            raise ValueError("max_seq_len must be positive and dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class TinyTransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if token_ids.ndim != 2 or token_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("token_ids must have shape (batch, sequence) within max_seq_len")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(token_ids.shape[1], token_ids.shape[1], device=token_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.transformer(hidden, mask=causal_mask)
        logits = self.output(self.norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return logits, loss