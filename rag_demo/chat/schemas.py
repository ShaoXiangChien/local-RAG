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


@dataclass
class InferenceMetrics:
    model_name: str | None = None
    estimated_prompt_tokens: int | None = None
    included_source_count: int = 0
    answer_chunk_count: int = 0
    thinking_chunk_count: int = 0
    answer_character_count: int = 0
    time_to_first_token_ms: int | None = None
    generation_elapsed_ms: int | None = None
    total_turn_elapsed_ms: int | None = None
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration_ns: int | None = None
    eval_count: int | None = None
    eval_duration_ns: int | None = None
    prompt_tokens_per_second: float | None = None
    output_tokens_per_second: float | None = None
    estimated_model_parameter_count: int | None = None
    estimated_effective_gflops: float | None = None


@dataclass(frozen=True)
class ChatAskResult:
    stream: Iterator[str]
    sources: list[SourceChunk]
    rewrite: object
    prompt: PromptBuildResult
    retrieval_query: str
    performance_metrics: InferenceMetrics = field(default_factory=InferenceMetrics)
    summary_status: dict[str, str] = field(default_factory=dict)
