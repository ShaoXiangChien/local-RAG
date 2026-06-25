from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    filename: str
    source_path: str
    content_hash: str
    file_type: str
    size_bytes: int
    created_at: datetime
    updated_at: datetime
    chunk_count: int
    embedding_model: str
    status: str
    error_message: str | None


class DocumentManifest:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    embedding_model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash)"
            )

    def upsert(self, record: DocumentRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    doc_id, filename, source_path, content_hash, file_type, size_bytes,
                    created_at, updated_at, chunk_count, embedding_model, status, error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    filename=excluded.filename,
                    source_path=excluded.source_path,
                    content_hash=excluded.content_hash,
                    file_type=excluded.file_type,
                    size_bytes=excluded.size_bytes,
                    updated_at=excluded.updated_at,
                    chunk_count=excluded.chunk_count,
                    embedding_model=excluded.embedding_model,
                    status=excluded.status,
                    error_message=excluded.error_message
                """,
                _record_to_row(record),
            )

    def list_documents(self, include_deleted: bool = False) -> list[DocumentRecord]:
        query = "SELECT * FROM documents"
        params: tuple[str, ...] = ()
        if not include_deleted:
            query += " WHERE status != ?"
            params = ("deleted",)
        query += " ORDER BY updated_at DESC, filename ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_record(row) for row in rows]

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return _row_to_record(row) if row else None

    def get_by_hash(self, content_hash: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE content_hash = ? AND status != ? ORDER BY updated_at DESC",
                (content_hash, "deleted"),
            ).fetchone()
        return _row_to_record(row) if row else None

    def mark_deleted(self, doc_id: str) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET status = ?, updated_at = ? WHERE doc_id = ?",
                ("deleted", now, doc_id),
            )

    def has_indexed_documents(self) -> bool:
        with self._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE status = ?", ("indexed",)
            ).fetchone()[0]
        return bool(count)

    def embedding_model_mismatch(self, current_embedding_model: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) FROM documents
                WHERE status = ? AND embedding_model != ?
                """,
                ("indexed", current_embedding_model),
            ).fetchone()
        return bool(row and row[0])


def _record_to_row(record: DocumentRecord) -> tuple[object, ...]:
    return (
        record.doc_id,
        record.filename,
        record.source_path,
        record.content_hash,
        record.file_type,
        record.size_bytes,
        record.created_at.isoformat(),
        record.updated_at.isoformat(),
        record.chunk_count,
        record.embedding_model,
        record.status,
        record.error_message,
    )


def _row_to_record(row: tuple[object, ...]) -> DocumentRecord:
    return DocumentRecord(
        doc_id=str(row[0]),
        filename=str(row[1]),
        source_path=str(row[2]),
        content_hash=str(row[3]),
        file_type=str(row[4]),
        size_bytes=int(row[5]),
        created_at=datetime.fromisoformat(str(row[6])),
        updated_at=datetime.fromisoformat(str(row[7])),
        chunk_count=int(row[8]),
        embedding_model=str(row[9]),
        status=str(row[10]),
        error_message=str(row[11]) if row[11] is not None else None,
    )
