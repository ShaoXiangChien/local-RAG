from rag_demo.chat.schemas import SourceChunk
from rag_demo.chat.token_budget import ContextBudgetConfig, HeuristicTokenCounter, TokenBudgeter


def _source(index: int, words: int) -> SourceChunk:
    return SourceChunk(
        chunk_id=f"doc:{index}",
        doc_id="doc",
        filename="notes.md",
        chunk_index=index,
        text=" ".join(f"w{index}_{i}" for i in range(words)),
        score=1.0 - (index / 100),
        start_char=0,
        end_char=words,
        token_count_estimate=words,
    )


def test_token_budgeter_keeps_prompt_plus_reserved_response_under_limit() -> None:
    budgeter = TokenBudgeter(
        HeuristicTokenCounter(),
        ContextBudgetConfig(max_working_context_tokens=120, reserved_response_tokens=30),
    )

    packed = budgeter.pack_sources(
        system_prompt="system prompt",
        question="question",
        rewritten_query="rewritten query",
        memory_text="memory words",
        sources=[_source(1, 45), _source(2, 45), _source(3, 45)],
    )

    assert packed.total_tokens + packed.reserved_response_tokens <= 120
    assert len(packed.sources) < 3
    assert packed.dropped_source_count > 0


def test_token_budgeter_truncates_summary_before_dropping_all_sources() -> None:
    budgeter = TokenBudgeter(
        HeuristicTokenCounter(),
        ContextBudgetConfig(
            max_working_context_tokens=160,
            reserved_response_tokens=30,
            reserved_memory_summary_tokens=20,
        ),
    )

    packed = budgeter.pack_sources(
        system_prompt="system prompt",
        question="question",
        rewritten_query="rewritten query",
        memory_text=" ".join(f"memory{i}" for i in range(80)),
        sources=[_source(1, 25)],
    )

    assert "memory79" not in packed.memory_text
    assert packed.sources
