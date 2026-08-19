"""
Splits a Document's raw text into Chunks small enough to embed and retrieve
individually.

Strategy: recursive character split. Try to cut on paragraph breaks first;
only fall back to sentence or word boundaries for a piece that's still too
big after that. This keeps chunks aligned with the document's actual
structure instead of slicing mid-sentence, which is what a blind fixed-size
split would do.

Sizes are in characters, not tokens — there's no tokenizer wired up for the
chat model, so CHUNK_SIZE/CHUNK_OVERLAP use the common ~4-chars-per-token
approximation (2000 chars ≈ 512 tokens, 200 chars ≈ 50 tokens/10% overlap).
# ponytail: character-count approximation, not exact token count. Swap in a
# real tokenizer (e.g. the model's own) if exact sizing ever matters.
"""
from dataclasses import dataclass

if __package__ in (None, ""):
    # Allows running this file directly (`python src/chunker/chunker.py`)
    # in addition to importing it as part of the `src` package.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.loader.loader import Document
else:
    from ..loader.loader import Document

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    text: str
    source: str  # which document this chunk came from, for citation
    chunk_index: int  # position within that document


def _split(text: str, separators: list[str]) -> list[str]:
    """Break text into pieces on the first separator, recursing into any
    piece that's still bigger than CHUNK_SIZE with the remaining separators."""
    sep = separators[0]
    pieces = text.split(sep) if sep else list(text)
    if len(separators) == 1:
        return [p for p in pieces if p]
    result = []
    for piece in pieces:
        if not piece:
            continue
        if len(piece) > CHUNK_SIZE:
            result.extend(_split(piece, separators[1:]))
        else:
            result.append(piece)
    return result


def _merge(pieces: list[str]) -> list[str]:
    """Greedily pack split pieces back together up to CHUNK_SIZE, carrying
    the tail of each chunk into the next one as overlap."""
    chunks = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            if current:
                chunks.append(current)
            overlap = current[-CHUNK_OVERLAP:] if current else ""
            current = f"{overlap} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_document(doc: Document) -> list[Chunk]:
    pieces = _split(doc.text, SEPARATORS)
    texts = _merge(pieces)
    return [Chunk(text=t, source=doc.source, chunk_index=i) for i, t in enumerate(texts)]


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc))
    return chunks


if __name__ == "__main__":
    # Self-check: chunk the real corpus and report sizes per document.
    # Run with: .venv/Scripts/python.exe src/chunker/chunker.py
    from src.loader.loader import load_documents

    docs = load_documents("data/knowledge")
    assert docs, "no documents found in data/knowledge — run the loader first"

    all_chunks = chunk_documents(docs)
    assert all_chunks, "chunking produced no chunks"

    by_source = {}
    for c in all_chunks:
        by_source.setdefault(c.source, []).append(c)

    for source, chunks in by_source.items():
        sizes = [len(c.text) for c in chunks]
        print(f"{source}: {len(chunks)} chunks, sizes {min(sizes)}-{max(sizes)} chars")

    oversized = [c for c in all_chunks if len(c.text) > CHUNK_SIZE * 1.5]
    assert not oversized, f"{len(oversized)} chunks are way over CHUNK_SIZE — separators aren't working"
    print(f"\nTotal: {len(all_chunks)} chunks from {len(docs)} documents")
