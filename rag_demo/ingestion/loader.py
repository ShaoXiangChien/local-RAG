from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


SUPPORTED_EXTENSIONS = {".md", ".txt"}


class UnsupportedFileTypeError(ValueError):
    pass


class DocumentDecodeError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedDocument:
    filename: str
    source_path: Path
    file_type: str
    text: str
    content_hash: str
    size_bytes: int


def load_text_document(path: str | Path) -> LoadedDocument:
    source_path = Path(path)
    file_type = source_path.suffix.lower()
    if file_type not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(f"Unsupported file type '{file_type}'. Use .md or .txt.")

    try:
        raw_text = source_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentDecodeError(f"Could not decode '{source_path.name}' as UTF-8.") from exc

    text = normalize_text(raw_text)
    return LoadedDocument(
        filename=source_path.name,
        source_path=source_path,
        file_type=file_type,
        text=text,
        content_hash=sha256(text.encode("utf-8")).hexdigest(),
        size_bytes=source_path.stat().st_size,
    )


def normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()
