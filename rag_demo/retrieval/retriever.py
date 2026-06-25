from __future__ import annotations

from rag_demo.embeddings.base import Embedder


class HybridRetriever:
    def __init__(self, embedder: Embedder, store, mode: str = "hybrid", top_k: int = 8):
        self.embedder = embedder
        self.store = store
        self.mode = mode
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None):
        if not query.strip():
            return []
        query_embedding = self.embedder.embed_query(query)
        return self.store.search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k or self.top_k,
            mode=self.mode,
        )
