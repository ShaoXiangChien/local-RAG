from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMStreamChunk:
    kind: str
    text: str


class LocalLLMClient(Protocol):
    def healthcheck(self) -> dict[str, Any]:
        """Return local model runtime health details."""

    def complete_chat(
        self, messages: list[dict[str, str]], options: dict[str, Any] | None = None
    ) -> str:
        """Return one complete chat response."""

    def complete_prompt(self, prompt: str, options: dict[str, Any] | None = None) -> str:
        """Return one complete response for a plain prompt."""

    def stream_chat(
        self, messages: list[dict[str, str]], options: dict[str, Any] | None = None
    ) -> Iterator[str]:
        """Yield a local chat response token by token."""
