from __future__ import annotations

from dataclasses import dataclass

from rag_demo.chat.token_budget import HeuristicTokenCounter
from rag_demo.models.base import LocalLLMClient


@dataclass(frozen=True)
class SummaryUpdateResult:
    summary: str
    error: str | None = None


class ConversationSummarizer:
    def __init__(
        self,
        llm: LocalLLMClient,
        counter: HeuristicTokenCounter | None = None,
        summary_token_budget: int = 600,
        enabled: bool = True,
    ):
        self.llm = llm
        self.counter = counter or HeuristicTokenCounter()
        self.summary_token_budget = summary_token_budget
        self.enabled = enabled

    def update_summary(
        self, previous_summary: str, question: str, answer: str
    ) -> SummaryUpdateResult:
        if not self.enabled:
            return SummaryUpdateResult(previous_summary)
        messages = [
            {
                "role": "system",
                "content": (
                    "Update a compact conversation summary for a local RAG assistant. "
                    "Preserve user goals, constraints, selected tools, unresolved questions, "
                    "and important entities. Do not include long source excerpts."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Previous summary:\n{previous_summary or 'None'}\n\n"
                    f"Latest user question:\n{question}\n\n"
                    f"Latest assistant answer:\n{answer}\n\n"
                    f"Keep the summary under {self.summary_token_budget} tokens."
                ),
            },
        ]
        try:
            summary = self.llm.complete_chat(
                messages,
                options={"temperature": 0.0, "num_predict": 160},
            ).strip()
        except Exception as exc:
            return SummaryUpdateResult(previous_summary, error=str(exc))
        summary = self.counter.truncate(summary, self.summary_token_budget)
        return SummaryUpdateResult(summary or previous_summary)
