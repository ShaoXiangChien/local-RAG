from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PipelineEvent:
    step: str
    status: str
    message: str
    elapsed_ms: int | None = None
    detail: str | None = None


@dataclass(frozen=True)
class SourceChunk:
    chunk_id: str
    doc_id: str
    filename: str
    chunk_index: int
    text: str
    score: float
    start_char: int
    end_char: int
    token_count_estimate: int
    content_hash: str | None = None

    @property
    def excerpt(self) -> str:
        compact = " ".join(self.text.split())
        return compact[:280] + ("..." if len(compact) > 280 else "")


@dataclass(frozen=True)
class ChatTurn:
    role: str
    content: str


@dataclass(frozen=True)
class PromptBuildResult:
    messages: list[dict[str, str]]
    sources: list[SourceChunk]
    memory_text: str
    dropped_source_count: int
    total_prompt_tokens: int


@dataclass(frozen=True)
class ChatAskResult:
    stream: Iterator[str]
    sources: list[SourceChunk]
    rewrite: object
    prompt: PromptBuildResult
    retrieval_query: str
    summary_status: dict[str, str] = field(default_factory=dict)
