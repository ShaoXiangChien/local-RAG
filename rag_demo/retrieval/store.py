from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rag_demo.chat.schemas import SourceChunk
from rag_demo.ingestion.chunker import TextChunk


class LanceDBVectorStore:
    """Local LanceDB wrapper with hybrid search and dense fallback."""

    def __init__(self, index_dir: str | Path, table_name: str = "chunks", full_text_index: bool = True):
        self.index_dir = Path(index_dir)
        self.table_name = table_name
        self.full_text_index = full_text_index
        self._db = None

    def _connect(self):
        if self._db is None:
            try:
                import lancedb
            except ImportError as exc:  # pragma: no cover - dependency/environment dependent
                raise RuntimeError("Install 'lancedb' to use the LanceDB vector store.") from exc
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.index_dir))
        return self._db

    def _table(self):
        db = self._connect()
        try:
            return db.open_table(self.table_name)
        except Exception:
            return None

    def upsert_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
        content_hash: str,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return

        rows = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "filename": chunk.filename,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "embedding": list(map(float, embedding)),
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "token_count_estimate": chunk.token_count_estimate,
                "content_hash": content_hash,
            }
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        db = self._connect()
        table = self._table()
        if table is None:
            table = db.create_table(self.table_name, data=rows)
        else:
            for doc_id in {chunk.doc_id for chunk in chunks}:
                self.delete_document(doc_id)
            table.add(rows)

        if self.full_text_index:
            try:
                table.create_fts_index("text", replace=True)
            except TypeError:
                table.create_fts_index("text")
            except Exception:
                pass

    def delete_document(self, doc_id: str) -> None:
        table = self._table()
        if table is None:
            return
        escaped = doc_id.replace("'", "''")
        try:
            table.delete(f"doc_id = '{escaped}'")
        except Exception:
            pass

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 8,
        mode: str = "hybrid",
    ) -> list[SourceChunk]:
        table = self._table()
        if table is None:
            return []

        rows: list[dict[str, Any]] = []
        if mode == "hybrid" and self.full_text_index:
            rows = self._hybrid_search(table, query, query_embedding, top_k)
        if not rows:
            rows = self._dense_search(table, query_embedding, top_k)
        return [_row_to_source(row) for row in rows]

    def _hybrid_search(
        self, table: Any, query: str, query_embedding: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        builders: list[Any] = []
        try:
            builders.append(
                table.search(query_type="hybrid")
                .vector(query_embedding)
                .text(query)
                .limit(top_k)
            )
        except Exception:
            pass
        try:
            builders.append(
                table.search(
                    query,
                    query_type="hybrid",
                    vector_column_name="embedding",
                    fts_columns="text",
                ).limit(top_k)
            )
        except Exception:
            pass

        for builder in builders:
            try:
                return _to_records(builder)
            except Exception:
                continue
        return []

    def _dense_search(
        self, table: Any, query_embedding: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        if not query_embedding:
            return []
        try:
            return _to_records(
                table.search(query_embedding, vector_column_name="embedding").limit(top_k)
            )
        except TypeError:
            return _to_records(table.search(query_embedding).limit(top_k))


def _to_records(builder: Any) -> list[dict[str, Any]]:
    try:
        return list(builder.to_list())
    except AttributeError:
        df = builder.to_pandas()
        return [dict(row) for row in df.to_dict("records")]


def _row_to_source(row: dict[str, Any]) -> SourceChunk:
    score = row.get("_relevance_score")
    if score is None:
        score = row.get("_score")
    if score is None and row.get("_distance") is not None:
        score = 1.0 / (1.0 + float(row["_distance"]))
    return SourceChunk(
        chunk_id=str(row["chunk_id"]),
        doc_id=str(row["doc_id"]),
        filename=str(row["filename"]),
        chunk_index=int(row["chunk_index"]),
        text=str(row["text"]),
        score=float(score if score is not None else 0.0),
        start_char=int(row.get("start_char", 0)),
        end_char=int(row.get("end_char", 0)),
        token_count_estimate=int(row.get("token_count_estimate", 0)),
        content_hash=str(row["content_hash"]) if row.get("content_hash") else None,
    )
