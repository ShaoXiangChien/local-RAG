from __future__ import annotations

from rag_demo.chat.memory import ConversationMemory
from rag_demo.chat.schemas import PromptBuildResult, SourceChunk
from rag_demo.chat.token_budget import TokenBudgeter


SYSTEM_PROMPT = """You are a local RAG assistant. Answer only the latest user question using only the provided context.
If the context is insufficient, say that the documents do not contain enough information.
Do not invent sources. Cite source chunk IDs when making factual claims.
Start with the answer. Do not include hidden reasoning, step-by-step analysis, or preamble.
Avoid phrases such as "let me analyze", "I need to", or "looking through the context".
Conversation memory is only for resolving references; do not answer an older question from memory unless the latest question asks for it.
Keep the answer concise and clear."""


class PromptBuilder:
    def __init__(self, budgeter: TokenBudgeter | None = None):
        self.budgeter = budgeter or TokenBudgeter()

    def build(
        self,
        question: str,
        rewritten_query: str,
        memory: ConversationMemory,
        sources: list[SourceChunk],
    ) -> PromptBuildResult:
        packed = self.budgeter.pack_sources(
            system_prompt=SYSTEM_PROMPT,
            question=question,
            rewritten_query=rewritten_query,
            memory_text=memory.as_text(),
            sources=sources,
        )
        user_content = self._render_user_content(
            question=question,
            rewritten_query=rewritten_query,
            memory_text=packed.memory_text,
            sources=packed.sources,
        )
        return PromptBuildResult(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            sources=packed.sources,
            memory_text=packed.memory_text,
            dropped_source_count=packed.dropped_source_count,
            total_prompt_tokens=packed.total_tokens,
        )

    def _render_user_content(
        self,
        question: str,
        rewritten_query: str,
        memory_text: str,
        sources: list[SourceChunk],
    ) -> str:
        source_text = "\n\n".join(_format_source(source) for source in sources)
        if not source_text:
            source_text = "No retrieved source chunks were available."
        memory_block = memory_text or "No prior conversation memory."
        return (
            f"Question to answer now:\n{question}\n\n"
            f"Standalone retrieval query used:\n{rewritten_query}\n\n"
            "Conversation memory (for reference only; do not answer old questions):\n"
            f"{memory_block}\n\n"
            f"Retrieved context:\n{source_text}\n\n"
            f"Answer only this latest question: {question}"
        )


def _format_source(source: SourceChunk) -> str:
    return (
        f"[source: {source.filename} | chunk: {source.chunk_index} | "
        f"id: {source.chunk_id} | score: {source.score:.3f}]\n"
        f"{source.text}"
    )
