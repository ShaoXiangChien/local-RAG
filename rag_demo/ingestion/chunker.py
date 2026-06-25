from __future__ import annotations

import re
from dataclasses import dataclass


TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_token_size: int = 450
    chunk_token_overlap: int = 70


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    doc_id: str
    filename: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    token_count_estimate: int


class RecursiveChunker:
    """Simple token-window chunker with overlap and source offsets."""

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()
        if self.config.chunk_token_overlap >= self.config.chunk_token_size:
            raise ValueError("chunk_token_overlap must be smaller than chunk_token_size")

    def split(self, text: str, doc_id: str, filename: str) -> list[TextChunk]:
        spans = list(TOKEN_RE.finditer(text))
        if not spans:
            return []

        chunks: list[TextChunk] = []
        size = self.config.chunk_token_size
        overlap = self.config.chunk_token_overlap
        step = max(1, size - overlap)
        token_start = 0

        while token_start < len(spans):
            token_end = min(token_start + size, len(spans))
            start_char = spans[token_start].start()
            end_char = spans[token_end - 1].end()
            chunk_text = text[start_char:end_char].strip()
            chunk_index = len(chunks)
            chunks.append(
                TextChunk(
                    chunk_id=f"{doc_id}:{chunk_index}",
                    doc_id=doc_id,
                    filename=filename,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    token_count_estimate=_estimate_tokens(chunk_text),
                )
            )
            if token_end == len(spans):
                break
            token_start += step

        return chunks


def estimate_tokens(text: str) -> int:
    return _estimate_tokens(text)


def _estimate_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))
