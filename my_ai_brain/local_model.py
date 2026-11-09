"""Optional adapter for locally stored Hugging Face-compatible fine-tuned models."""

from typing import Protocol


class TextGenerator(Protocol):
    def generate(self, prompt: str, *, max_new_tokens: int = 40, **options: object) -> str: ...


class HuggingFaceLocalBackend:
    def __init__(self, model: object, tokenizer: object, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_pretrained(cls, model_directory: str, device: str = "cpu") -> "HuggingFaceLocalBackend":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Install the optional 'transformers' package to use this backend") from error
        tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_directory, local_files_only=True, trust_remote_code=False
        ).to(device)
        return cls(model, tokenizer, device)

    def generate(self, prompt: str, *, max_new_tokens: int = 40, **options: object) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            generation_options = {"max_new_tokens": max_new_tokens, "do_sample": True, **options}
            generation_options.setdefault("pad_token_id", self.tokenizer.eos_token_id)
            output = self.model.generate(**inputs, **generation_options)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)