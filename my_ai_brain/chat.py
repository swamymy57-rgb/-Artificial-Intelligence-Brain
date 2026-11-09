"""Memory-aware chat helpers and a local interactive CLI."""

import argparse
from pathlib import Path

from my_ai_brain.artifacts import load_artifacts
from my_ai_brain.config import Settings
from my_ai_brain.inference import InferenceEngine
from my_ai_brain.local_model import TextGenerator
from my_ai_brain.memory import SQLiteMemory


def chat(
    engine: TextGenerator, memory: SQLiteMemory, message: str, max_new_tokens: int = 40
) -> dict[str, object]:
    memories = memory.search(message, limit=3)
    memory_context = "\n".join(str(item["content"])[:500] for item in memories)
    prompt = f"Relevant memory:\n{memory_context}\nUser: {message}\nAssistant:"
    generated = engine.generate(prompt, max_new_tokens=max_new_tokens)
    reply = generated[len(prompt) :] if generated.startswith(prompt) else generated
    memory.add(f"User: {message}")
    return {"reply": reply, "memories": memories}


def main() -> None:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Chat with the locally trained MY-AI-BRAIN model")
    parser.add_argument("--model-dir", type=Path, default=Path(settings.model_dir))
    parser.add_argument("--database", type=Path, default=Path(settings.database))
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--message", help="Send one message and exit; omit for interactive chat")
    arguments = parser.parse_args()

    model, tokenizer = load_artifacts(arguments.model_dir, settings.device)
    engine = InferenceEngine(model, tokenizer, settings.device)
    memory = SQLiteMemory(arguments.database)

    def respond(message: str) -> None:
        result = chat(engine, memory, message, arguments.max_new_tokens)
        for item in result["memories"]:
            print(f"Memory: {item['content']}")
        print(f"AI: {result['reply']}")

    if arguments.message is not None:
        respond(arguments.message)
        return
    print("Local chat ready. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            message = input("You: ").strip()
        except EOFError:
            break
        if message.casefold() in {"exit", "quit"}:
            break
        if message:
            respond(message)


if __name__ == "__main__":
    main()
