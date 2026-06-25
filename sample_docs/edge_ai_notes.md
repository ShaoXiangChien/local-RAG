# Edge AI RAG Notes

The prototype is designed for local-first edge AI constraints. Documents, prompts, embeddings, vector search, and answer generation stay on the developer machine.

## Runtime Choice

Ollama was selected because it provides a scriptable local model runtime, a Python client, streaming chat responses, local embeddings, and context options such as `num_ctx`.

## Retrieval Choice

LanceDB was selected because it runs as an embedded local vector store and supports vector search, full-text search, and hybrid retrieval. Hybrid retrieval helps technical documents because exact terms such as `num_ctx`, `embeddinggemma`, and filenames matter alongside semantic similarity.

## Context Budget

The application enforces a 6,000-token working-context budget before calling the model. It reserves response space, packs conversation memory, and includes only retrieved chunks that fit.

## Error Handling

If query rewrite fails, the app falls back to the original user question. If the local model server is unavailable, the UI shows a clear warning. If no documents are indexed, chat asks the user to upload documents first.
