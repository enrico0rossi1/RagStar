"""
Sends assembled chat messages to the LLM and returns its answer text.

Temperature 0.0: deterministic output, so eval runs are reproducible and
comparable across pipeline changes. This does NOT guarantee correctness —
it only removes sampling randomness as a source of variation. Groundedness
depends on retrieval quality and the strict-grounding prompt (see
src/prompt/prompt.py); this module just executes the LLM call.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = OpenAI(base_url=os.environ["BASE_URL"], api_key="ollama")
_MODEL = os.environ["CHAT_MODEL"]

TEMPERATURE = 0.0
MAX_TOKENS = 512


@dataclass
class GenerationResult:
    text: str
    completion_tokens: int  # for tokens/sec throughput measurement in eval.py


def generate(messages: list[dict]) -> GenerationResult:
    response = _client.chat.completions.create(
        model=_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return GenerationResult(
        text=response.choices[0].message.content,
        completion_tokens=response.usage.completion_tokens,
    )


if __name__ == "__main__":
    # Self-check: send a trivial prompt and confirm we get an answer back.
    # Run with: .venv/Scripts/python.exe src/generator/generator.py
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Reply with exactly the word: pong"},
    ]
    result = generate(messages)
    assert result.text, "got an empty response"
    assert result.completion_tokens > 0
    print(f"Model replied: {result.text!r} ({result.completion_tokens} tokens)")
