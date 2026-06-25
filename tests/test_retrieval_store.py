from rag_demo.ingestion.chunker import TextChunk
from rag_demo.retrieval.store import LanceDBVectorStore


def test_lancedb_store_upserts_searches_and_deletes_chunks(tmp_path) -> None:
    store = LanceDBVectorStore(tmp_path / "index", full_text_index=True)
    chunk = TextChunk(
        chunk_id="doc1:0",
        doc_id="doc1",
        filename="notes.md",
        chunk_index=0,
        text="Ollama runs the language model locally.",
        start_char=0,
        end_char=44,
        token_count_estimate=7,
    )

    store.upsert_chunks([chunk], [[1.0, 0.0, 0.0]], content_hash="hash")
    results = store.search(
        query="Ollama local runtime",
        query_embedding=[1.0, 0.0, 0.0],
        top_k=3,
        mode="hybrid",
    )

    assert [result.chunk_id for result in results] == ["doc1:0"]

    store.delete_document("doc1")

    assert store.search("Ollama", [1.0, 0.0, 0.0], top_k=3, mode="hybrid") == []
