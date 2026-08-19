"""
Indexing pipeline: load documents, chunk them, embed each chunk, persist to
a local vector store. Run once per corpus update:

    .venv/Scripts/python.exe src/ingest.py

Rebuilds the vector store from scratch every run (drop + recreate the
table) instead of incrementally upserting. Simplest way to guarantee
re-running never duplicates chunks.
# ponytail: full rebuild, not incremental upsert. Fine at this corpus size
# (hundreds of chunks, seconds to re-embed). Upgrade to incremental
# indexing (e.g. keyed by a content hash, only re-embed changed chunks) if
# the corpus grows large enough that re-embedding everything gets slow.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lancedb

from src.chunker.chunker import chunk_documents
from src.embedder.embedder import embed
from src.loader.loader import load_documents

KNOWLEDGE_DIR = "data/knowledge"
VECTOR_DB_DIR = "vector_db"
TABLE_NAME = "chunks"
EMBED_BATCH_SIZE = 32  # chunks per embedding API call


def ingest(knowledge_dir: str = KNOWLEDGE_DIR) -> int:
    """Load, chunk, embed, and store every document in `knowledge_dir`.

    Returns the number of chunks stored.
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
                }
            )
        print(f"  embedded {len(records)}/{len(chunks)}")

    db = lancedb.connect(VECTOR_DB_DIR)
    db.create_table(TABLE_NAME, data=records, mode="overwrite")
    print(f"Stored {len(records)} chunks in {VECTOR_DB_DIR}/{TABLE_NAME}")
    return len(records)


if __name__ == "__main__":
    ingest()
