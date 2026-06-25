from rag_demo.chat.memory import ConversationMemory
from rag_demo.chat.prompt import PromptBuilder
from rag_demo.chat.schemas import ChatTurn, SourceChunk
from rag_demo.chat.token_budget import ContextBudgetConfig, HeuristicTokenCounter, TokenBudgeter


def test_prompt_builder_returns_messages_and_exact_sources_used_in_prompt() -> None:
    sources = [
        SourceChunk(
            chunk_id="doc:0",
            doc_id="doc",
            filename="notes.md",
            chunk_index=0,
            text="The project uses Ollama locally.",
            score=0.9,
            start_char=0,
            end_char=32,
            token_count_estimate=6,
        )
    ]
    memory = ConversationMemory(
        summary="The user is evaluating a local RAG prototype.",
        recent_turns=[
            ChatTurn(role="user", content="What stack did we pick?"),
            ChatTurn(role="assistant", content="Ollama, LanceDB, Streamlit."),
        ],
    )
    builder = PromptBuilder(
        TokenBudgeter(
            HeuristicTokenCounter(),
            ContextBudgetConfig(max_working_context_tokens=400, reserved_response_tokens=80),
        )
    )

    prompt = builder.build(
        question="Why Ollama?",
        rewritten_query="why use Ollama local RAG",
        memory=memory,
        sources=sources,
    )

    assert prompt.sources == sources
    assert prompt.messages[0]["role"] == "system"
    assert "using only the provided context" in prompt.messages[0]["content"]
    assert "Do not include hidden reasoning" in prompt.messages[0]["content"]
    assert "Start with the answer" in prompt.messages[0]["content"]
    user_content = prompt.messages[-1]["content"]
    assert user_content.startswith(
        "Question to answer now:\nWhy Ollama?\n\n"
        "Standalone retrieval query used:\nwhy use Ollama local RAG"
    )
    assert "Conversation memory (for reference only; do not answer old questions):" in user_content
    assert "[source: notes.md | chunk: 0 | id: doc:0" in user_content
    assert "Answer only this latest question: Why Ollama?" in user_content
