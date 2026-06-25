from rag_demo.chat.memory import ConversationMemory
from rag_demo.chat.query_rewrite import QueryRewriteService
from rag_demo.chat.schemas import ChatTurn
from rag_demo.models.base import LocalLLMClient


class FakeRewriteLLM(LocalLLMClient):
    def __init__(self, response: str | Exception):
        self.response = response
        self.messages = None
        self.prompt = None
        self.options = None

    def healthcheck(self):
        return {"ok": True}

    def complete_chat(self, messages, options=None):
        self.messages = messages
        self.options = options
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def complete_prompt(self, prompt, options=None):
        self.prompt = prompt
        self.options = options
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def stream_chat(self, messages, options=None):
        yield self.complete_chat(messages, options)


def test_query_rewrite_skips_llm_when_no_memory_context() -> None:
    llm = FakeRewriteLLM("should not be called")
    service = QueryRewriteService(llm, enabled=True)

    result = service.rewrite("What Ollama endpoint should embeddings use?", ConversationMemory())

    assert result.rewritten_query == "What Ollama endpoint should embeddings use?"
    assert result.rewrite_used is False
    assert result.rewrite_error == "No conversation memory available for query rewrite."
    assert llm.messages is None


def test_query_rewrite_returns_standalone_query_from_llm() -> None:
    llm = FakeRewriteLLM(" local-only RAG context budget ")
    service = QueryRewriteService(llm, enabled=True)

    result = service.rewrite("What about the limit?", ConversationMemory(summary="6k token limit"))

    assert result.rewritten_query == "local-only RAG context budget"
    assert result.rewrite_used is True
    assert result.rewrite_error is None
    assert "Do not answer" in llm.prompt
    assert llm.messages is None
    assert llm.options == {
        "temperature": 0.0,
        "num_predict": 64,
        "think": False,
        "stop": ["\n"],
    }


def test_query_rewrite_resolves_what_about_followup_before_calling_llm() -> None:
    llm = FakeRewriteLLM("wrong old embedding query")
    service = QueryRewriteService(llm, enabled=True)
    memory = ConversationMemory(
        recent_turns=[
            ChatTurn(
                role="user",
                content="What Ollama endpoint should this app use for embeddings?",
            ),
            ChatTurn(role="assistant", content="Use POST /api/embed."),
        ]
    )

    result = service.rewrite("what about chat?", memory)

    assert result.rewritten_query == "What Ollama endpoint should this app use for chat?"
    assert result.rewrite_used is True
    assert result.rewrite_error is None
    assert llm.messages is None


def test_query_rewrite_falls_back_to_original_on_empty_or_error() -> None:
    empty = QueryRewriteService(FakeRewriteLLM(""), enabled=True).rewrite(
        "What about it?", ConversationMemory(summary="The app uses Ollama.")
    )
    failed = QueryRewriteService(FakeRewriteLLM(RuntimeError("offline")), enabled=True).rewrite(
        "What about it?", ConversationMemory(summary="The app uses Ollama.")
    )

    assert empty.rewritten_query == "What about it?"
    assert empty.rewrite_used is False
    assert failed.rewritten_query == "What about it?"
    assert failed.rewrite_used is False
    assert "offline" in failed.rewrite_error
