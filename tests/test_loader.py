from pathlib import Path

import pytest

from rag_demo.ingestion.loader import (
    DocumentDecodeError,
    LoadedDocument,
    UnsupportedFileTypeError,
    load_text_document,
)


def test_load_text_document_accepts_markdown_and_normalizes_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# Title\r\n\r\nA line.   \r\n", encoding="utf-8")

    loaded = load_text_document(path)

    assert isinstance(loaded, LoadedDocument)
    assert loaded.filename == "notes.md"
    assert loaded.file_type == ".md"
    assert loaded.text == "# Title\n\nA line."
    assert loaded.content_hash
    assert loaded.size_bytes == path.stat().st_size


def test_load_text_document_accepts_plain_text(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("plain text", encoding="utf-8")

    loaded = load_text_document(path)

    assert loaded.file_type == ".txt"
    assert loaded.text == "plain text"


def test_load_text_document_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "notes.pdf"
    path.write_bytes(b"%PDF")

    with pytest.raises(UnsupportedFileTypeError):
        load_text_document(path)


def test_load_text_document_reports_utf8_decode_errors(tmp_path: Path) -> None:
    path = tmp_path / "broken.txt"
    path.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(DocumentDecodeError):
        load_text_document(path)
