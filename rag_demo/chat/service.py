from __future__ import annotations

from collections.abc import Iterator
from time import perf_counter
from typing import Callable, TypeVar

from rag_demo.chat.memory import ConversationMemoryBuffer
from rag_demo.chat.prompt import PromptBuilder
from rag_demo.chat.query_rewrite import QueryRewriteService
from rag_demo.chat.schemas import ChatAskResult, ChatTurn, PipelineEvent
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
        stream = self._stream_and_update_memory(
            session_id=session_id,
            question=question,
            previous_summary=memory.summary,
            messages=prompt.messages,
            on_event=on_event,
            on_thinking=on_thinking,
        )
        return ChatAskResult(
            stream=stream,
            sources=prompt.sources,
            rewrite=rewrite,
            prompt=prompt,
            retrieval_query=rewrite.rewritten_query,
        )

    def _stream_and_update_memory(
        self,
        session_id: str,
        question: str,
        previous_summary: str,
        messages: list[dict[str, str]],
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
                if chunk.kind == "thinking":
                    thinking_parts.append(chunk.text)
                    if on_thinking is not None:
                        on_thinking(chunk.text)
                    continue

                if chunk.text and not first_token_seen:
                    first_token_seen = True
                    _emit(
                        on_event,
                        "generation",
                        "first_token",
                        "First visible token received",
                        _elapsed_ms(generation_start),
                    )
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
        _emit(
            on_event,
            "generation",
            "completed",
            "Stream answer from local model",
            _elapsed_ms(generation_start),
            f"{len(parts)} answer chunks streamed; {len(thinking_parts)} thinking chunks separated",
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
