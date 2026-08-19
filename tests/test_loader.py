from src.loader.loader import load_documents


def test_loads_txt_and_md(tmp_path):
    (tmp_path / "a.txt").write_text("Plain text file about foxes.", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Heading\n\nSome content.", encoding="utf-8")

    docs = load_documents(str(tmp_path))

    assert {d.source for d in docs} == {"a.txt", "b.md"}
    assert all(d.text.strip() for d in docs)


def test_skips_unsupported_extension(tmp_path):
    (tmp_path / "a.txt").write_text("kept", encoding="utf-8")
    (tmp_path / "c.docx").write_text("not supported", encoding="utf-8")

    docs = load_documents(str(tmp_path))

    assert [d.source for d in docs] == ["a.txt"]


def test_skips_empty_file(tmp_path):
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
    (tmp_path / "real.txt").write_text("content", encoding="utf-8")

    docs = load_documents(str(tmp_path))

    assert [d.source for d in docs] == ["real.txt"]
