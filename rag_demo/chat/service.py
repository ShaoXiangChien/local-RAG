from __future__ import annotations

from collections.abc import Iterator
import re
from time import perf_counter
from typing import Any, Callable, TypeVar

from rag_demo.chat.memory import ConversationMemoryBuffer
from rag_demo.chat.prompt import PromptBuilder
from rag_demo.chat.query_rewrite import QueryRewriteService
from rag_demo.chat.schemas import ChatAskResult, ChatTurn, InferenceMetrics, PipelineEvent
from rag_demo.chat.summarizer import ConversationSummarizer
from rag_demo.models.base import LLMStreamChunk, LocalLLMClient

PipelineEventHandler = Callable[[PipelineEvent], None]
ThinkingEventHandler = Callable[[str], None]
T = TypeVar("T")


class ChatService:
    def __init__(
        self,
        llm: LocalLLMClient,
        retriever,
        query_rewriter: QueryRewriteService,
        prompt_builder: PromptBuilder,
        memory_buffer: ConversationMemoryBuffer,
        summarizer: ConversationSummarizer,
        top_k: int = 8,
        generation_options: dict | None = None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.query_rewriter = query_rewriter
        self.prompt_builder = prompt_builder
        self.memory_buffer = memory_buffer
        self.summarizer = summarizer
        self.top_k = top_k
        self.generation_options = generation_options or {}

    def ask(
        self,
        session_id: str,
        question: str,
        on_event: PipelineEventHandler | None = None,
        on_thinking: ThinkingEventHandler | None = None,
    ) -> ChatAskResult:
        request_start = perf_counter()
        memory = _run_step(
            on_event,
            "memory",
            "Load conversation memory",
            lambda: self.memory_buffer.load(session_id),
            detail=lambda result: (
                f"{len(result.recent_turns)} recent turns; "
                f"summary {'present' if result.summary else 'empty'}"
            ),
        )
        rewrite = _run_step(
            on_event,
            "query_rewrite",
            "Rewrite the question for retrieval",
            lambda: self.query_rewriter.rewrite(question, memory),
            detail=lambda result: (
                result.rewrite_error
                if result.rewrite_error
                else f"using: {result.rewritten_query}"
            ),
        )
        sources = _run_step(
            on_event,
            "retrieval",
            "Embed query and search the local index",
            lambda: self.retriever.retrieve(rewrite.rewritten_query, top_k=self.top_k),
            detail=lambda result: f"{len(result)} source chunks returned",
        )
        prompt = _run_step(
            on_event,
            "prompt",
            "Build grounded prompt within token budget",
            lambda: self.prompt_builder.build(
                question=question,
                rewritten_query=rewrite.rewritten_query,
                memory=memory,
                sources=sources,
            ),
            detail=lambda result: (
                f"{result.total_prompt_tokens} estimated prompt tokens; "
                f"{len(result.sources)} included sources"
            ),
        )
        metrics = InferenceMetrics(
            model_name=str(getattr(self.llm, "model", "")) or None,
            estimated_prompt_tokens=prompt.total_prompt_tokens,
            included_source_count=len(prompt.sources),
        )
        stream = self._stream_and_update_memory(
            session_id=session_id,
            question=question,
            previous_summary=memory.summary,
            messages=prompt.messages,
            performance_metrics=metrics,
            request_start=request_start,
            on_event=on_event,
            on_thinking=on_thinking,
        )
        return ChatAskResult(
            stream=stream,
            sources=prompt.sources,
            rewrite=rewrite,
            prompt=prompt,
            retrieval_query=rewrite.rewritten_query,
            performance_metrics=metrics,
        )

    def _stream_and_update_memory(
        self,
        session_id: str,
        question: str,
        previous_summary: str,
        messages: list[dict[str, str]],
        performance_metrics: InferenceMetrics,
        request_start: float,
        on_event: PipelineEventHandler | None = None,
        on_thinking: ThinkingEventHandler | None = None,
    ) -> Iterator[str]:
        parts: list[str] = []
        thinking_parts: list[str] = []
        generation_start = perf_counter()
        first_token_seen = False
        _emit(on_event, "generation", "started", "Stream answer from local model")
        try:
            for chunk in _stream_llm_chunks(
                self.llm,
                messages,
                options=self.generation_options,
            ):
                if chunk.kind == "metrics":
                    _apply_ollama_metrics(
                        performance_metrics,
                        chunk.metadata or {},
                        performance_metrics.model_name,
                    )
                    continue
                if chunk.kind == "thinking":
                    thinking_parts.append(chunk.text)
                    performance_metrics.thinking_chunk_count += 1
                    if on_thinking is not None:
                        on_thinking(chunk.text)
                    continue

                if not chunk.text:
                    continue

                if not first_token_seen:
                    first_token_seen = True
                    performance_metrics.time_to_first_token_ms = _elapsed_ms(generation_start)
                    _emit(
                        on_event,
                        "generation",
                        "first_token",
                        "First visible token received",
                        _elapsed_ms(generation_start),
                    )
                performance_metrics.answer_chunk_count += 1
                performance_metrics.answer_character_count += len(chunk.text)
                parts.append(chunk.text)
                yield chunk.text
        except Exception as exc:
            _emit(
                on_event,
                "generation",
                "failed",
                "Stream answer from local model",
                _elapsed_ms(generation_start),
                str(exc),
            )
            raise
        performance_metrics.generation_elapsed_ms = _elapsed_ms(generation_start)
        _emit(
            on_event,
            "generation",
            "completed",
            "Stream answer from local model",
            performance_metrics.generation_elapsed_ms,
            _generation_detail(performance_metrics, len(parts), len(thinking_parts)),
        )

        answer = "".join(parts)

        def update_turns() -> None:
            self.memory_buffer.add_turn(session_id, ChatTurn(role="user", content=question))
            self.memory_buffer.add_turn(session_id, ChatTurn(role="assistant", content=answer))

        _run_step(
            on_event,
            "memory_update",
            "Save chat turn to local memory",
            update_turns,
        )
        summary = _run_step(
            on_event,
            "summary",
            "Refresh compact conversation summary",
            lambda: self.summarizer.update_summary(previous_summary, question, answer),
            detail=lambda result: result.error or "summary updated",
        )
        if summary.summary:
            self.memory_buffer.save_summary(session_id, summary.summary)
        performance_metrics.total_turn_elapsed_ms = _elapsed_ms(request_start)


def _run_step(
    on_event: PipelineEventHandler | None,
    step: str,
    message: str,
    action: Callable[[], T],
    detail: Callable[[T], str] | None = None,
) -> T:
    started_at = perf_counter()
    _emit(on_event, step, "started", message)
    try:
        result = action()
    except Exception as exc:
        _emit(on_event, step, "failed", message, _elapsed_ms(started_at), str(exc))
        raise
    _emit(
        on_event,
        step,
        "completed",
        message,
        _elapsed_ms(started_at),
        detail(result) if detail else None,
    )
    return result


def _emit(
    on_event: PipelineEventHandler | None,
    step: str,
    status: str,
    message: str,
    elapsed_ms: int | None = None,
    detail: str | None = None,
) -> None:
    if on_event is None:
        return
    on_event(
        PipelineEvent(
            step=step,
            status=status,
            message=message,
            elapsed_ms=elapsed_ms,
            detail=detail,
        )
    )


def _elapsed_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1000))


def _stream_llm_chunks(
    llm: LocalLLMClient,
    messages: list[dict[str, str]],
    options: dict | None = None,
) -> Iterator[LLMStreamChunk]:
    stream_chat_events = getattr(llm, "stream_chat_events", None)
    if callable(stream_chat_events):
        yield from stream_chat_events(messages, options=options)
        return

    for token in llm.stream_chat(messages, options=options):
        yield LLMStreamChunk(kind="content", text=token)


def _apply_ollama_metrics(
    metrics: InferenceMetrics,
    metadata: dict[str, Any],
    model_name: str | None,
) -> None:
    metrics.total_duration_ns = _int_or_none(metadata.get("total_duration_ns"))
    metrics.load_duration_ns = _int_or_none(metadata.get("load_duration_ns"))
    metrics.prompt_eval_count = _int_or_none(metadata.get("prompt_eval_count"))
    metrics.prompt_eval_duration_ns = _int_or_none(metadata.get("prompt_eval_duration_ns"))
    metrics.eval_count = _int_or_none(metadata.get("eval_count"))
    metrics.eval_duration_ns = _int_or_none(metadata.get("eval_duration_ns"))
    metrics.prompt_tokens_per_second = _tokens_per_second(
        metrics.prompt_eval_count,
        metrics.prompt_eval_duration_ns,
    )
    metrics.output_tokens_per_second = _tokens_per_second(
        metrics.eval_count,
        metrics.eval_duration_ns,
    )
    metrics.estimated_model_parameter_count = _estimate_model_parameter_count(model_name)
    if (
        metrics.output_tokens_per_second is not None
        and metrics.estimated_model_parameter_count is not None
    ):
        metrics.estimated_effective_gflops = round(
            metrics.output_tokens_per_second
            * 2
            * metrics.estimated_model_parameter_count
            / 1_000_000_000,
            2,
        )


def _tokens_per_second(count: int | None, duration_ns: int | None) -> float | None:
    if count is None or duration_ns is None or duration_ns <= 0:
        return None
    return round(count / (duration_ns / 1_000_000_000), 2)


def _estimate_model_parameter_count(model_name: str | None) -> int | None:
    if not model_name:
        return None
    match = re.search(r":(?P<size>\d+(?:\.\d+)?)b\b", model_name.lower())
    if not match:
        match = re.search(r"\b(?P<size>\d+(?:\.\d+)?)b\b", model_name.lower())
    if not match:
        return None
    return int(float(match.group("size")) * 1_000_000_000)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _generation_detail(
    metrics: InferenceMetrics,
    answer_chunk_count: int,
    thinking_chunk_count: int,
) -> str:
    pieces = [
        f"{answer_chunk_count} answer chunks streamed",
        f"{thinking_chunk_count} thinking chunks separated",
    ]
    if metrics.time_to_first_token_ms is not None:
        pieces.append(f"TTFT {metrics.time_to_first_token_ms} ms")
    if metrics.output_tokens_per_second is not None:
        pieces.append(f"{metrics.output_tokens_per_second:.2f} output tok/s")
    return "; ".join(pieces)
