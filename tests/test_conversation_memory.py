from rag_demo.chat.memory import ConversationMemoryBuffer
from rag_demo.chat.schemas import ChatTurn
from rag_demo.storage.conversation_store import ConversationStore


def test_conversation_store_persists_summary_by_session(tmp_path) -> None:
    store = ConversationStore(tmp_path / "app.db")

    store.save_summary("session-1", "The user cares about local-only inference.")

    assert store.load_summary("session-1") == "The user cares about local-only inference."
    assert store.load_summary("missing") == ""


def test_memory_buffer_keeps_only_recent_turns_and_loads_summary(tmp_path) -> None:
    store = ConversationStore(tmp_path / "app.db")
    store.save_summary("session-1", "Running summary")
    buffer = ConversationMemoryBuffer(store, recent_turns=2)

    buffer.add_turn("session-1", ChatTurn(role="user", content="one"))
    buffer.add_turn("session-1", ChatTurn(role="assistant", content="two"))
    buffer.add_turn("session-1", ChatTurn(role="user", content="three"))

    memory = buffer.load("session-1")

    assert memory.summary == "Running summary"
    assert [turn.content for turn in memory.recent_turns] == ["two", "three"]
