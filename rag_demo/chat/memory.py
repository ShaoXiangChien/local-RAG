from __future__ import annotations

from dataclasses import dataclass, field

from rag_demo.chat.schemas import ChatTurn
from rag_demo.storage.conversation_store import ConversationStore


@dataclass(frozen=True)
class ConversationMemory:
    summary: str = ""
    recent_turns: list[ChatTurn] = field(default_factory=list)

    def as_text(self) -> str:
        parts: list[str] = []
        if self.summary:
            parts.append(f"Conversation summary:\n{self.summary}")
        if self.recent_turns:
            turns = "\n".join(f"{turn.role.title()}: {turn.content}" for turn in self.recent_turns)
            parts.append(f"Recent turns:\n{turns}")
        return "\n\n".join(parts)


class ConversationMemoryBuffer:
    def __init__(self, store: ConversationStore, recent_turns: int = 2):
        self.store = store
        self.recent_turns = recent_turns
        self._turns_by_session: dict[str, list[ChatTurn]] = {}

    def load(self, session_id: str) -> ConversationMemory:
        turns = self._turns_by_session.get(session_id, [])
        return ConversationMemory(
            summary=self.store.load_summary(session_id),
            recent_turns=list(turns[-self.recent_turns :]),
        )

    def add_turn(self, session_id: str, turn: ChatTurn) -> None:
        turns = self._turns_by_session.setdefault(session_id, [])
        turns.append(turn)
        max_turns = max(self.recent_turns * 3, self.recent_turns)
        if len(turns) > max_turns:
            del turns[: len(turns) - max_turns]

    def save_summary(self, session_id: str, summary: str) -> None:
        self.store.save_summary(session_id, summary)

    def reset(self, session_id: str) -> None:
        self._turns_by_session.pop(session_id, None)
        self.store.save_summary(session_id, "")
