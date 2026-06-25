from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from rag_demo.embeddings.base import Embedder
from rag_demo.ingestion.chunker import RecursiveChunker
from rag_demo.ingestion.loader import LoadedDocument, load_text_document
from rag_demo.storage.manifest import DocumentManifest, DocumentRecord


@dataclass(frozen=True)
class IngestionResult:
    doc_id: str
    filename: str
    chunk_count: int
    status: str
    message: str


class DocumentIngestionPipeline:
    def __init__(
        self,
        chunker: RecursiveChunker,
        embedder: Embedder,
        vector_store,
        manifest: DocumentManifest,
        upload_dir: str | Path,
        embedding_model: str,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.manifest = manifest
        self.upload_dir = Path(upload_dir)
        self.embedding_model = embedding_model
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def ingest_path(self, path: str | Path) -> IngestionResult:
        loaded = load_text_document(path)
        duplicate = self.manifest.get_by_hash(loaded.content_hash)
        if duplicate and duplicate.embedding_model == self.embedding_model:
            return IngestionResult(
                doc_id=duplicate.doc_id,
                filename=duplicate.filename,
                chunk_count=duplicate.chunk_count,
                status="skipped",
                message="Document content is already indexed.",
            )

        doc_id = make_doc_id(loaded)
        stored_path = self._copy_to_uploads(loaded.source_path, doc_id, loaded.file_type)
        chunks = self.chunker.split(loaded.text, doc_id=doc_id, filename=loaded.filename)
        embeddings = self.embedder.embed_documents([chunk.text for chunk in chunks])
        self.vector_store.upsert_chunks(chunks, embeddings, loaded.content_hash)

        now = datetime.now(UTC)
        self.manifest.upsert(
            DocumentRecord(
                doc_id=doc_id,
                filename=loaded.filename,
                source_path=str(stored_path),
                content_hash=loaded.content_hash,
                file_type=loaded.file_type,
                size_bytes=loaded.size_bytes,
                created_at=now,
                updated_at=now,
                chunk_count=len(chunks),
                embedding_model=self.embedding_model,
                status="indexed",
                error_message=None,
            )
        )
        return IngestionResult(
            doc_id=doc_id,
            filename=loaded.filename,
            chunk_count=len(chunks),
            status="indexed",
            message=f"Indexed {len(chunks)} chunks.",
        )

    def ingest_upload(self, filename: str, data: bytes) -> IngestionResult:
        target = self.upload_dir / filename
        target.write_bytes(data)
        return self.ingest_path(target)

    def delete_document(self, doc_id: str) -> None:
        record = self.manifest.get(doc_id)
        self.vector_store.delete_document(doc_id)
        self.manifest.mark_deleted(doc_id)
        if record:
            try:
                Path(record.source_path).unlink(missing_ok=True)
            except OSError:
                pass

    def _copy_to_uploads(self, source_path: Path, doc_id: str, file_type: str) -> Path:
        target = self.upload_dir / f"{doc_id}{file_type}"
        if source_path.resolve() != target.resolve():
            shutil.copy2(source_path, target)
        return target


def make_doc_id(document: LoadedDocument) -> str:
    return sha256(f"{document.filename}:{document.content_hash}".encode("utf-8")).hexdigest()[:24]
