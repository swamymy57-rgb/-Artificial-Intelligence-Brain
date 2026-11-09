# MY-AI-BRAIN

A small, local-first hybrid AI prototype. It includes a character tokenizer, a tiny causal Transformer built with PyTorch, a training command, local text generation, SQLite memory, and a FastAPI service. It does not call paid AI APIs and makes no claim of AGI or autonomous self-improvement.

## Requirements

- Python 3.10 or newer
- PyTorch, FastAPI, and the packages in `requirements.txt`
- Optional: a local Hugging Face-compatible model and the `transformers` package to use the separate fine-tuned-model adapter

Python 3.14 may not yet be supported by every PyTorch wheel in all environments. If installation fails, use a Python version supported by the current PyTorch release (Python 3.11 or 3.12 is a reliable choice).

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU-only PyTorch is sufficient for the prototype and avoids CUDA packages.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
cp .env.example .env
```

For GPU acceleration, install the PyTorch build appropriate for your hardware from the official PyTorch installation selector before installing the remaining requirements.

`.env` is for local settings and is ignored by Git. The app reads environment variables directly; it does not automatically parse `.env`. Export variables in your shell, or load `.env` with your preferred local environment manager. No secrets are required by the prototype.

## Train and evaluate the sample model

The repository includes a tiny UTF-8 English/Telugu dialogue corpus at `data/sample_telugu_english.txt`. Training is from scratch and local; this character-level model is a learning/demo artifact, not a capable general-purpose assistant.

Run the reproducible CPU training command from the repository root:

```bash
python -m my_ai_brain.train --text-file data/sample_telugu_english.txt --output-dir artifacts --epochs 5 --batch-size 8 --steps-per-epoch 10 --seed 7
```

The command deterministically splits the records into training and validation portions, trains on CPU by default, and reports final training loss plus validation loss, perplexity, and next-character accuracy. With the command above, output has this form (metric values are printed from the completed run):

```text
Saved model and checkpoints to artifacts
Train loss: 3.6045
Validation: loss=3.7110, perplexity=40.8957, accuracy=0.1944
```

These are the observed CPU output values for the command above; a different PyTorch/hardware version may vary slightly.

`artifacts/` contains `model.pt`, `model_config.json`, `tokenizer.json`, `metrics.json`, `training_config.json`, and `checkpoints/epoch-001.pt` through `epoch-005.pt`. Training data and artifacts stay on disk and are not uploaded. The seed fixes data splitting, initialization, dropout, and sampling for reproducible CPU runs.

For a quick run, use `--epochs 1 --batch-size 2 --steps-per-epoch 1`. To train on your own UTF-8 text, pass `--text-file path/to/your-corpus.txt`; use material you have permission to use. The input should have multiple non-empty lines/records, or at least four characters if it is a single line.

## Chat locally

After training, start the interactive local chat:

```bash
python -m my_ai_brain.chat --model-dir artifacts --database data/memory.sqlite3
```

Type messages and enter `exit` or `quit` to end. For a one-message inference:

```bash
python -m my_ai_brain.chat --model-dir artifacts --database data/memory.sqlite3 --message "Hello" --max-new-tokens 12
```

The CLI prints a generated `AI:` reply and any matching prior memory. Chat stores user messages in SQLite; retrieval uses simple local lexical matching, not semantic/vector search. The toy model can produce incoherent text, and memory matches are shown separately so retrieved facts are visible.

## Run the local API

Set paths if they differ from the defaults, then bind to loopback:

```bash
export MY_AI_BRAIN_MODEL_DIR=artifacts
export MY_AI_BRAIN_DATABASE=data/memory.sqlite3
uvicorn my_ai_brain.api:app --host 127.0.0.1 --port 8000
```

`GET http://127.0.0.1:8000/health` returns `{"status":"ok","model_loaded":true}` when artifacts loaded. `POST /chat` accepts `{"message":"Hello","max_new_tokens":12}` and returns a generated `reply` plus matching prior `memories`; the message is added to SQLite for later requests. `POST /generate` remains available for raw generation. `POST /memory` and `GET /memory?query=...` provide direct memory access. Generation and chat return `503` when no trained artifacts are loaded. The interactive API schema is at `http://127.0.0.1:8000/docs`.

Example requests:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' -d '{"message":"Please remember my favorite color is blue.","max_new_tokens":12}'
curl -s -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' -d '{"message":"What is my favorite color?","max_new_tokens":12}'
```

The health response is `{"status":"ok","model_loaded":true}`. The first chat response has an empty `memories` list; the second includes the prior `User: Please remember my favorite color is blue.` entry. Generated reply text and memory timestamps are runtime-dependent.

This is a local demo, not a deployment recipe. It intentionally has no authentication layer; do not expose it directly to an untrusted network. No paid API or external inference service is used.

## Local fine-tuned model adapter

The optional adapter in `my_ai_brain/local_model.py` loads a Hugging Face-compatible model from a local directory only. Install `transformers` separately, then construct `HuggingFaceLocalBackend.from_pretrained("/path/to/model")`. It disables remote code and remote downloads. This is a separate interface from the tiny built-in model, not a required dependency or a paid service.

## Architecture and extension points

- `tokenizer.py`, `model.py`, and `training.py` contain the small from-scratch language-model path.
- `inference.py` provides a framework-light generation interface usable by the API or another host.
- `memory.py` provides persistent SQLite text storage and simple search; it is not a vector database.
- `api.py` hosts local generation, chat, and memory endpoints behind FastAPI.
- `local_model.py` adapts an optional local fine-tuned model behind the same generation protocol.

Android on-device inference is a future integration, not implemented here. The separated tokenizer, model, and inference boundaries, plus the saved state-dict/config artifacts, provide a starting point for evaluating a mobile runtime and converting/exporting a compatible model. No Android compatibility or performance is claimed yet. This demo is not deployed; no external AI API is required.

## Tests and smoke check

```bash
python -m pytest
python scripts/smoke_test.py
```

The smoke script performs a one-epoch training run in a temporary directory, invokes the chat CLI against its saved checkpoint, checks API health, and verifies memory retrieval over separate chat requests. It prints `Smoke test passed: training, CLI inference, API health, and persistent chat memory.` on success.

## Configuration and data safety

Settings use the `MY_AI_BRAIN_` environment prefix. See `.env.example` for names and defaults. Never commit credentials or private training data. SQLite memory and model artifacts are local files; protect their filesystem permissions and backups. Only load model artifacts and corpora you trust. PyTorch state dicts are loaded with `weights_only=True`.