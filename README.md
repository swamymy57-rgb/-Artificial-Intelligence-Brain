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

## Train a model

Provide a UTF-8 text file you have permission to use. The model and tokenizer are trained from scratch; this is a toy character-level model, so quality depends heavily on the corpus and it is not a general-purpose assistant.

```bash
python -m my_ai_brain.train --text-file data/corpus.txt --output-dir artifacts --epochs 20
```

The command saves `model.pt`, `model_config.json`, and `tokenizer.json`. Training runs on CPU by default and uses a small model suitable for experimentation. The corpus is read locally and is not uploaded.

## Run the API

Set the artifact and database paths if they differ from the defaults, then start the server:

```bash
export MY_AI_BRAIN_MODEL_DIR=artifacts
export MY_AI_BRAIN_DATABASE=data/memory.sqlite3
uvicorn my_ai_brain.api:app --host 127.0.0.1 --port 8000
```

The server binds to loopback by default when run with the command above. Visit `http://127.0.0.1:8000/docs` for the local API schema. `GET /health` reports whether model artifacts loaded. `POST /generate` accepts `{"prompt":"...","max_new_tokens":40}`. `POST /memory` stores a text entry, and `GET /memory?query=...` searches entries. Generation returns `503` until artifacts are trained and available.

For deployment, put the API behind a trusted TLS-terminating reverse proxy, use network access controls, and choose a protected persistent database path. The prototype intentionally has no authentication layer; do not expose it directly to an untrusted network.

## Local fine-tuned model adapter

The optional adapter in `my_ai_brain/local_model.py` loads a Hugging Face-compatible model from a local directory only. Install `transformers` separately, then construct `HuggingFaceLocalBackend.from_pretrained("/path/to/model")`. It disables remote code and remote downloads. This is a separate interface from the tiny built-in model, not a required dependency or a paid service.

## Architecture and extension points

- `tokenizer.py`, `model.py`, and `training.py` contain the small from-scratch language-model path.
- `inference.py` provides a framework-light generation interface usable by the API or another host.
- `memory.py` provides persistent SQLite text storage and simple search; it is not a vector database.
- `api.py` hosts cloud inference behind FastAPI. It can also be embedded in a privately controlled deployment.
- `local_model.py` adapts an optional local fine-tuned model behind the same generation protocol.

Android on-device inference is a future integration, not implemented here. The separated tokenizer, model, and inference boundaries, plus the saved state-dict/config artifacts, provide a starting point for evaluating a mobile runtime and converting/exporting a compatible model. No Android compatibility or performance is claimed yet. Cloud inference means running this API on infrastructure you control; no external AI API is required.

## Tests

```bash
python -m pytest
```

## Configuration and data safety

Settings use the `MY_AI_BRAIN_` environment prefix. See `.env.example` for names and defaults. Never commit credentials or private training data. SQLite memory and model artifacts are local files; protect their filesystem permissions and backups. Only load model artifacts and corpora you trust. PyTorch state dicts are loaded with `weights_only=True`.