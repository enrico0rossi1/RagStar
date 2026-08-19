from src.prompt.prompt import SYSTEM_PROMPT, assemble_prompt
from src.retriever.retriever import RetrievedChunk


def _chunk(text, source="doc.txt", index=0, distance=0.1):
    return RetrievedChunk(text=text, source=source, chunk_index=index, distance=distance)


def test_assemble_prompt_shape():
    messages = assemble_prompt("What is X?", [_chunk("X is a thing.")])

    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1]["role"] == "user"


def test_assemble_prompt_includes_query_and_sources():
    chunks = [_chunk("Fact one.", source="a.txt"), _chunk("Fact two.", source="b.txt")]

    content = assemble_prompt("my question", chunks)[1]["content"]

    assert "my question" in content
    assert "Fact one." in content
    assert "[Source: a.txt]" in content
    assert "Fact two." in content
    assert "[Source: b.txt]" in content


def test_assemble_prompt_preserves_chunk_order():
    chunks = [_chunk("first", source="1.txt"), _chunk("second", source="2.txt")]

    content = assemble_prompt("q", chunks)[1]["content"]

    assert content.index("first") < content.index("second")
