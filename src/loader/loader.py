"""
Loads raw text out of source documents (.txt, .md, .pdf) so the chunker can
split it into pieces the embedder can turn into vectors.

Each file becomes one Document: its raw text plus which file it came from.
The source filename gets threaded through the rest of the pipeline (chunk
metadata, prompt citations) so a retrieved chunk can always be traced back
to where it came from.
"""
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Document:
    source: str  # filename, used for citation and debugging
    text: str


def _load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_pdf_file(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n\n".join(pages)


def load_documents(directory: str) -> list[Document]:
    """Load every supported file directly inside `directory` into a Document.

    A bad file (unsupported extension, corrupt PDF, no extractable text)
    is skipped with a printed warning instead of crashing the whole ingest
    run over one file.
    """
    documents = []
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"skipping {path.name}: unsupported extension")
            continue
        try:
            if path.suffix.lower() == ".pdf":
                text = _load_pdf_file(path)
            else:
                text = _load_text_file(path)
        except Exception as e:
            print(f"skipping {path.name}: failed to read ({e})")
            continue
        if not text.strip():
            print(f"skipping {path.name}: no extractable text")
            continue
        documents.append(Document(source=path.name, text=text))
    return documents


if __name__ == "__main__":
    # Self-check: point at a directory (default data/knowledge/, the corpus
    # that actually gets indexed — see data/other/ for anything that
    # shouldn't be ingested) and print what got loaded. Run with:
    #   .venv/Scripts/python.exe src/loader/loader.py [directory]
    import sys

    directory = sys.argv[1] if len(sys.argv) > 1 else "data/knowledge"
    docs = load_documents(directory)
    assert docs, f"no documents loaded from {directory} — add some files first"
    for doc in docs:
        print(f"{doc.source}: {len(doc.text)} chars")
