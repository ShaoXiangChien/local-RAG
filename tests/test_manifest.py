from datetime import UTC, datetime

from rag_demo.storage.manifest import DocumentManifest, DocumentRecord


def _record(doc_id: str = "doc1") -> DocumentRecord:
    now = datetime(2026, 6, 25, tzinfo=UTC)
    return DocumentRecord(
        doc_id=doc_id,
        filename="notes.md",
        source_path="/tmp/notes.md",
        content_hash="abc",
        file_type=".md",
        size_bytes=123,
        created_at=now,
        updated_at=now,
        chunk_count=2,
        embedding_model="embeddinggemma",
        status="indexed",
        error_message=None,
    )


def test_manifest_can_add_list_fetch_and_delete_records(tmp_path) -> None:
    manifest = DocumentManifest(tmp_path / "app.db")

    manifest.upsert(_record())
    listed = manifest.list_documents()

    assert [doc.doc_id for doc in listed] == ["doc1"]
    assert manifest.get_by_hash("abc").doc_id == "doc1"

    manifest.mark_deleted("doc1")

    assert manifest.get("doc1").status == "deleted"
    assert manifest.list_documents(include_deleted=False) == []


def test_manifest_detects_embedding_model_mismatch(tmp_path) -> None:
    manifest = DocumentManifest(tmp_path / "app.db")
    manifest.upsert(_record())

    assert manifest.embedding_model_mismatch("other-model")
    assert not manifest.embedding_model_mismatch("embeddinggemma")
