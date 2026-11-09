import torch

from my_ai_brain.artifacts import load_artifacts, save_artifacts
from my_ai_brain.inference import InferenceEngine
from my_ai_brain.model import ModelConfig, TinyTransformerLM
from my_ai_brain.tokenizer import CharacterTokenizer
from my_ai_brain.training import evaluate_model, train_model


def test_tokenizer_round_trip_and_unknown_character():
    tokenizer = CharacterTokenizer.train("hello")
    assert tokenizer.decode(tokenizer.encode("hello")) == "hello"
    assert tokenizer.encode("?") == [0]
    assert tokenizer.decode([0]) == ""


def test_transformer_logits_loss_and_training():
    tokenizer = CharacterTokenizer.train("tiny transformer")
    model = TinyTransformerLM(ModelConfig(vocab_size=tokenizer.vocab_size, d_model=16, n_heads=2, n_layers=1))
    inputs = torch.tensor([[1, 2, 3]])
    logits, loss = model(inputs, torch.tensor([[2, 3, 4]]))
    assert logits.shape == (1, 3, tokenizer.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    losses = train_model(model, tokenizer, "tiny transformer" * 4, epochs=1, batch_size=2, steps_per_epoch=1)
    assert len(losses) == 1 and losses[0] > 0


def test_evaluation_reports_finite_loss_perplexity_and_accuracy():
    tokenizer = CharacterTokenizer.train("evaluation text")
    model = TinyTransformerLM(ModelConfig(vocab_size=tokenizer.vocab_size, d_model=16, n_heads=2, n_layers=1))
    metrics = evaluate_model(model, tokenizer, "evaluation")
    assert set(metrics) == {"loss", "perplexity", "accuracy"}
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics["perplexity"] >= 1
    assert 0 <= metrics["accuracy"] <= 1


def test_training_saves_epoch_checkpoints(tmp_path):
    tokenizer = CharacterTokenizer.train("checkpoint test")
    model = TinyTransformerLM(ModelConfig(vocab_size=tokenizer.vocab_size, d_model=8, n_heads=2, n_layers=1))
    train_model(
        model, tokenizer, "checkpoint test " * 3, epochs=2, batch_size=2, steps_per_epoch=1,
        checkpoint_dir=tmp_path,
    )
    assert sorted(path.name for path in tmp_path.glob("*.pt")) == ["epoch-001.pt", "epoch-002.pt"]


def test_transformer_cannot_attend_to_future_tokens():
    model = TinyTransformerLM(ModelConfig(vocab_size=5, d_model=16, n_heads=2, n_layers=1, dropout=0))
    model.eval()
    first, _ = model(torch.tensor([[1, 2, 3]]))
    changed_future, _ = model(torch.tensor([[1, 2, 4]]))
    assert torch.allclose(first[:, :2], changed_future[:, :2], atol=1e-6)


def test_artifact_round_trip_and_generation(tmp_path):
    tokenizer = CharacterTokenizer.train("hello world")
    model = TinyTransformerLM(ModelConfig(vocab_size=tokenizer.vocab_size, d_model=16, n_heads=2, n_layers=1))
    save_artifacts(model, tokenizer, tmp_path)
    loaded_model, loaded_tokenizer = load_artifacts(tmp_path)
    engine = InferenceEngine(loaded_model, loaded_tokenizer)
    result = engine.generate("hello", max_new_tokens=2)
    assert result.startswith("hello")
    assert 5 <= len(result) <= 7