from __future__ import annotations

from typing import Any


class OllamaEmbedder:
    """Embedding adapter that keeps vectors behind the local Ollama runtime."""

    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        if "ollama.com" in host:
            raise ValueError("Cloud Ollama hosts are not allowed for this local-only demo.")
        self.model_name = model_name
        self.host = host
        self._client = None

    def _ollama_client(self):
        if self._client is None:
            try:
                from ollama import Client
            except ImportError as exc:
                raise RuntimeError("Install the 'ollama' Python package to use Ollama.") from exc
            self._client = Client(host=self.host)
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._ollama_client().embed(model=self.model_name, input=texts)
        return _extract_embeddings(response)

    def embed_query(self, text: str) -> list[float]:
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []


def _extract_embeddings(response: Any) -> list[list[float]]:
    if isinstance(response, dict):
        if "embeddings" in response:
            return [list(map(float, row)) for row in response["embeddings"]]
        if "embedding" in response:
            return [list(map(float, response["embedding"]))]
    embeddings = getattr(response, "embeddings", None)
    if embeddings is not None:
        return [list(map(float, row)) for row in embeddings]
    embedding = getattr(response, "embedding", None)
    if embedding is not None:
        return [list(map(float, embedding))]
    raise RuntimeError("Ollama did not return embeddings.")
