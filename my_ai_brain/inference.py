"""Framework-light autoregressive generation for compatible local models."""

import torch

from my_ai_brain.model import TinyTransformerLM
from my_ai_brain.tokenizer import CharacterTokenizer


class InferenceEngine:
    def __init__(self, model: TinyTransformerLM, tokenizer: CharacterTokenizer, device: str = "cpu"):
        if model.config.vocab_size != tokenizer.vocab_size:
            raise ValueError("Tokenizer vocabulary size does not match model configuration")
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 40,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> str:
        if max_new_tokens < 0 or temperature <= 0:
            raise ValueError("max_new_tokens must be non-negative and temperature must be positive")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive")
        generated = self.tokenizer.encode(prompt)
        if not generated:
            raise ValueError("Prompt must contain at least one character")
        output = list(generated)
        for _ in range(max_new_tokens):
            context = torch.tensor(
                [output[-self.model.config.max_seq_len :]], dtype=torch.long, device=self.device
            )
            logits, _ = self.model(context)
            next_logits = logits[0, -1] / temperature
            if top_k is not None:
                count = min(top_k, next_logits.numel())
                threshold = torch.topk(next_logits, count).values[-1]
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
            next_token = int(torch.multinomial(torch.softmax(next_logits, dim=-1), num_samples=1))
            output.append(next_token)
        return prompt + self.tokenizer.decode(output[len(generated) :])