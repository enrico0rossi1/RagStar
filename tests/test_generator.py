"""Requires Ollama running locally with qwen2.5:7b pulled."""
from src.generator.generator import generate


def test_generate_returns_text():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Reply with exactly the word: pong"},
    ]

    result = generate(messages)

    assert result.text.strip()
    assert result.completion_tokens > 0
