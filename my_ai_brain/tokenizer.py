"""A deterministic character-level tokenizer for tiny experiments."""

import json
from pathlib import Path


class CharacterTokenizer:
    def __init__(self, vocabulary: list[str]):
        if not vocabulary or vocabulary[0] != "<unk>" or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("Vocabulary must be unique and start with <unk>")
        self.vocabulary = vocabulary
        self._token_to_id = {token: index for index, token in enumerate(vocabulary)}

    @classmethod
    def train(cls, text: str) -> "CharacterTokenizer":
        return cls(["<unk>", *sorted(set(text))])

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    def encode(self, text: str) -> list[int]:
        unknown_id = self._token_to_id["<unk>"]
        return [self._token_to_id.get(character, unknown_id) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(
            self.vocabulary[token_id]
            for token_id in token_ids
            if 0 <= token_id < len(self.vocabulary) and token_id != 0
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"vocabulary": self.vocabulary}, ensure_ascii=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["vocabulary"])