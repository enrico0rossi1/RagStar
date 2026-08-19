from src.chunker.chunker import CHUNK_SIZE, chunk_document, chunk_documents
from src.loader.loader import Document


def test_short_text_stays_one_chunk():
    doc = Document(source="short.txt", text="A short sentence.")

    chunks = chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].text == "A short sentence."
    assert chunks[0].source == "short.txt"
    assert chunks[0].chunk_index == 0


def test_long_text_splits_into_multiple_bounded_chunks():
    paragraph = "word " * 100  # ~500 chars
    doc = Document(source="long.txt", text=(paragraph + "\n\n") * 20)  # ~10k chars

    chunks = chunk_document(doc)

    assert len(chunks) > 1
    assert all(len(c.text) <= CHUNK_SIZE * 1.5 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_documents_flattens_and_keeps_source():
    docs = [
        Document(source="a.txt", text="first doc"),
        Document(source="b.txt", text="second doc"),
    ]

    chunks = chunk_documents(docs)

    assert {c.source for c in chunks} == {"a.txt", "b.txt"}
