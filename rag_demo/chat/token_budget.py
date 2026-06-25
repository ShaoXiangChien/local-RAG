from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag_demo.chat.schemas import SourceChunk


TOKEN_RE = re.compile(r"\S+")


class HeuristicTokenCounter:
    def count(self, text: str) -> int:
        return len(TOKEN_RE.findall(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        matches = list(TOKEN_RE.finditer(text))
        if len(matches) <= max_tokens:
            return text
        return text[: matches[max_tokens - 1].end()].rstrip()


@dataclass(frozen=True)
class ContextBudgetConfig:
    max_working_context_tokens: int = 6000
    reserved_response_tokens: int = 800
    reserved_system_tokens: int = 300
    reserved_memory_summary_tokens: int = 600
    reserved_recent_turn_tokens: int = 500
    reserved_query_tokens: int = 350
    safety_margin_tokens: int = 50


@dataclass(frozen=True)
class PackedSources:
    sources: list[SourceChunk]
    memory_text: str
    total_tokens: int
    reserved_response_tokens: int
    dropped_source_count: int = 0
    diagnostics: dict[str, int] = field(default_factory=dict)


class TokenBudgeter:
    def __init__(
        self,
        counter: HeuristicTokenCounter | None = None,
        config: ContextBudgetConfig | None = None,
    ):
        self.counter = counter or HeuristicTokenCounter()
        self.config = config or ContextBudgetConfig()

    def pack_sources(
        self,
        system_prompt: str,
        question: str,
        rewritten_query: str,
        memory_text: str,
        sources: list[SourceChunk],
    ) -> PackedSources:
        cfg = self.config
        memory_budget = cfg.reserved_memory_summary_tokens
        if "Recent turns:" in memory_text:
            memory_budget += cfg.reserved_recent_turn_tokens
        packed_memory = self.counter.truncate(memory_text, memory_budget)

        fixed_text = "\n\n".join([system_prompt, question, rewritten_query, packed_memory])
        fixed_tokens = self.counter.count(fixed_text)
        prompt_limit = cfg.max_working_context_tokens - cfg.reserved_response_tokens
        usable_limit = max(0, prompt_limit - cfg.safety_margin_tokens)
        available_for_sources = max(0, usable_limit - fixed_tokens)

        packed_sources: list[SourceChunk] = []
        source_tokens = 0
        for source in sources:
            rendered = _render_source_for_count(source)
            candidate_tokens = self.counter.count(rendered)
            if source_tokens + candidate_tokens <= available_for_sources:
                packed_sources.append(source)
                source_tokens += candidate_tokens

        dropped = len(sources) - len(packed_sources)
        return PackedSources(
            sources=packed_sources,
            memory_text=packed_memory,
            total_tokens=fixed_tokens + source_tokens,
            reserved_response_tokens=cfg.reserved_response_tokens,
            dropped_source_count=dropped,
            diagnostics={
                "fixed_tokens": fixed_tokens,
                "source_tokens": source_tokens,
                "usable_limit": usable_limit,
            },
        )


def _render_source_for_count(source: SourceChunk) -> str:
    return f"[source: {source.filename} | chunk: {source.chunk_index}]\n{source.text}"
