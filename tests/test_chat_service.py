from rag_demo.chat.memory import ConversationMemoryBuffer
from rag_demo.chat.prompt import PromptBuilder
from rag_demo.chat.query_rewrite import QueryRewriteResult
from rag_demo.chat.schemas import SourceChunk
from rag_demo.chat.service import ChatService
from rag_demo.chat.summarizer import SummaryUpdateResult
from rag_demo.chat.token_budget import ContextBudgetConfig, HeuristicTokenCounter, TokenBudgeter
from rag_demo.models.base import LLMStreamChunk, LocalLLMClient
from rag_demo.storage.conversation_store import ConversationStore


class FakeStreamingLLM(LocalLLMClient):
    def __init__(self):
        self.messages = None

    def healthcheck(self):
        return {"ok": True}

    def complete_chat(self, messages, options=None):
        return "complete"

    def stream_chat(self, messages, options=None):
        self.messages = messages
        yield "Local "
        yield "answer"


class FakeThinkingLLM(FakeStreamingLLM):
    def stream_chat_events(self, messages, options=None):
        self.messages = messages
        yield LLMStreamChunk(kind="thinking", text="checking chunks")
        yield LLMStreamChunk(kind="content", text="Final ")
        yield LLMStreamChunk(kind="content", text="answer")


class FakeRewriter:
    def __init__(self):
        self.input_question = None

    def rewrite(self, question, memory):
        self.input_question = question
        return QueryRewriteResult(
            rewritten_query="standalone retrieval query",
            rewrite_used=True,
            rewrite_error=None,
        )


class FakeRetriever:
    def __init__(self):
        self.query = None

    def retrieve(self, query, top_k=8):
        self.query = query
        return [
            SourceChunk(
                chunk_id="doc:0",
                doc_id="doc",
                filename="notes.md",
                chunk_index=0,
                text="Use Ollama locally.",
                score=0.88,
                start_char=0,
                end_char=20,
                token_count_estimate=4,
            )
        ]


class FakeSummarizer:
    def __init__(self):
        self.updated = False

    def update_summary(self, previous_summary, question, answer):
        self.updated = True
        return SummaryUpdateResult(summary=f"{previous_summary} {question} {answer}".strip())


def test_chat_service_uses_rewritten_query_for_retrieval_and_original_question_for_prompt(tmp_path) -> None:
    llm = FakeStreamingLLM()
    rewriter = FakeRewriter()
    retriever = FakeRetriever()
    summarizer = FakeSummarizer()
    memory = ConversationMemoryBuffer(ConversationStore(tmp_path / "app.db"), recent_turns=2)
    service = ChatService(
        llm=llm,
        retriever=retriever,
        query_rewriter=rewriter,
        prompt_builder=PromptBuilder(
            TokenBudgeter(
                HeuristicTokenCounter(),
                ContextBudgetConfig(max_working_context_tokens=400, reserved_response_tokens=80),
            )
        ),
        memory_buffer=memory,
        summarizer=summarizer,
        top_k=4,
    )

    result = service.ask("session-1", "What about its local runtime?")
    answer = "".join(result.stream)

    assert answer == "Local answer"
    assert rewriter.input_question == "What about its local runtime?"
    assert retriever.query == "standalone retrieval query"
    assert "Question to answer now:\nWhat about its local runtime?" in llm.messages[-1]["content"]
    assert result.sources[0].chunk_id == "doc:0"
    assert result.rewrite.rewrite_used is True
    assert summarizer.updated is True
    assert "Local answer" in memory.load("session-1").summary


def test_chat_service_emits_pipeline_events(tmp_path) -> None:
    llm = FakeStreamingLLM()
    rewriter = FakeRewriter()
    retriever = FakeRetriever()
    summarizer = FakeSummarizer()
    memory = ConversationMemoryBuffer(ConversationStore(tmp_path / "app.db"), recent_turns=2)
    service = ChatService(
        llm=llm,
        retriever=retriever,
        query_rewriter=rewriter,
        prompt_builder=PromptBuilder(
            TokenBudgeter(
                HeuristicTokenCounter(),
                ContextBudgetConfig(max_working_context_tokens=400, reserved_response_tokens=80),
            )
        ),
        memory_buffer=memory,
        summarizer=summarizer,
        top_k=4,
    )
    events = []

    result = service.ask("session-1", "What about its local runtime?", on_event=events.append)

    assert [(event.step, event.status) for event in events] == [
        ("memory", "started"),
        ("memory", "completed"),
        ("query_rewrite", "started"),
        ("query_rewrite", "completed"),
        ("retrieval", "started"),
        ("retrieval", "completed"),
        ("prompt", "started"),
        ("prompt", "completed"),
    ]

    answer = "".join(result.stream)

    assert answer == "Local answer"
    assert (events[-1].step, events[-1].status) == ("summary", "completed")
    assert ("generation", "first_token") in [
        (event.step, event.status) for event in events
    ]
    assert all(event.elapsed_ms is None or event.elapsed_ms >= 0 for event in events)


def test_chat_service_keeps_thinking_out_of_visible_answer_and_memory(tmp_path) -> None:
    memory = ConversationMemoryBuffer(ConversationStore(tmp_path / "app.db"), recent_turns=2)
    service = ChatService(
        llm=FakeThinkingLLM(),
        retriever=FakeRetriever(),
        query_rewriter=FakeRewriter(),
        prompt_builder=PromptBuilder(
            TokenBudgeter(
                HeuristicTokenCounter(),
                ContextBudgetConfig(max_working_context_tokens=400, reserved_response_tokens=80),
            )
        ),
        memory_buffer=memory,
        summarizer=FakeSummarizer(),
        top_k=4,
    )
    thinking_tokens = []

    result = service.ask(
        "session-thinking",
        "What about chat?",
        on_thinking=thinking_tokens.append,
    )
    answer = "".join(result.stream)

    assert answer == "Final answer"
    assert thinking_tokens == ["checking chunks"]
    assert [turn.content for turn in memory.load("session-thinking").recent_turns] == [
        "What about chat?",
        "Final answer",
    ]
