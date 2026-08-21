"""
Indexing pipeline: load documents, chunk them, embed each chunk, persist to
a local vector store. Run once per corpus update:

    .venv/Scripts/python.exe src/ingest.py

Rebuilds the vector store from scratch every run (drop + recreate the
table) instead of incrementally upserting. Simplest way to guarantee
re-running never duplicates chunks.
# ponytail: full rebuild, not incremental upsert, single global table.
# Fine for one corpus at this size (hundreds of chunks, seconds to
# re-embed). Two upgrade triggers, independent of each other: (1) corpus
# grows large enough that re-embedding everything gets slow -> incremental
# indexing keyed by a content hash. (2) multiple users/datasets need to
# coexist (this becomes a shared upload tool, not a single local corpus)
# -> this overwrite-the-whole-table call is actively wrong, not just slow;
# needs per-collection tables or a collection_id column filtered at query
# time, decided before that use case shows up for real.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lancedb

from src.chunker.chunker import chunk_documents
from src.embedder.embedder import embed
from src.enrichment.enrichment import generate_questions
from src.loader.loader import load_documents

KNOWLEDGE_DIR = "data/knowledge"
VECTOR_DB_DIR = "vector_db"
TABLE_NAME = "chunks"
EMBED_BATCH_SIZE = 32  # chunks/questions per embedding API call


def ingest(knowledge_dir: str = KNOWLEDGE_DIR) -> int:
    """Load, chunk, embed, and store every document in `knowledge_dir`.

    Each chunk is stored once under its own embedding (kind="chunk") and
    again under the embedding of each hypothetical question it answers
    (kind="question", Reverse HyDE — see src/enrichment/enrichment.py).
    Both kinds carry the same chunk text, so retrieval always returns real
    chunk text regardless of which embedding matched.

    Returns the number of rows stored (chunks + questions).
    """
    docs = load_documents(knowledge_dir)
    if not docs:
        raise ValueError(f"no documents found in {knowledge_dir}")
    print(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    records = []
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        vectors = embed([c.text for c in batch])
        for chunk, vector in zip(batch, vectors):
            records.append(
                {
                    "vector": vector,
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "kind": "chunk",
                }
            )
        print(f"  embedded {len(records)}/{len(chunks)} chunks")

    question_texts, question_chunks = [], []
    for n, chunk in enumerate(chunks, start=1):
        for q in generate_questions(chunk.text):
            question_texts.append(q)
            question_chunks.append(chunk)
        print(f"  generated questions for {n}/{len(chunks)} chunks")

    for i in range(0, len(question_texts), EMBED_BATCH_SIZE):
        text_batch = question_texts[i : i + EMBED_BATCH_SIZE]
        chunk_batch = question_chunks[i : i + EMBED_BATCH_SIZE]
        vectors = embed(text_batch)
        for chunk, vector in zip(chunk_batch, vectors):
            records.append(
                {
                    "vector": vector,
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "kind": "question",
                }
            )
        print(f"  embedded {i + len(text_batch)}/{len(question_texts)} questions")

    db = lancedb.connect(VECTOR_DB_DIR)
    db.create_table(TABLE_NAME, data=records, mode="overwrite")
    print(f"Stored {len(records)} rows ({len(chunks)} chunks + {len(question_texts)} questions) in {VECTOR_DB_DIR}/{TABLE_NAME}")
    return len(records)


if __name__ == "__main__":
    ingest()
