from __future__ import annotations

import re
from dataclasses import dataclass

from rag_demo.chat.memory import ConversationMemory
from rag_demo.models.base import LocalLLMClient

FOLLOWUP_TOPIC_RE = re.compile(
    r"^\s*(?:and\s+)?(?:what|how)(?:'s|\s+is)?\s+about\s+(?P<topic>.+?)\s*\??\s*$",
    re.IGNORECASE,
)
TRAILING_TOPIC_RE = re.compile(
    r"\b(?P<lead>for|about|of|with|using|use for|used for)\s+[^?.,;]+(?P<punct>[?.!]?)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryRewriteResult:
    rewritten_query: str
    rewrite_used: bool
    rewrite_error: str | None = None


class QueryRewriteService:
    def __init__(self, llm: LocalLLMClient, enabled: bool = True):
        self.llm = llm
        self.enabled = enabled

    def rewrite(self, question: str, memory: ConversationMemory) -> QueryRewriteResult:
        if not self.enabled:
            return QueryRewriteResult(question, rewrite_used=False)
        if not memory.as_text():
            return QueryRewriteResult(
                question,
                rewrite_used=False,
                rewrite_error="No conversation memory available for query rewrite.",
            )
        deterministic = _rewrite_simple_followup(question, memory)
        if deterministic:
            return QueryRewriteResult(deterministic, rewrite_used=True)
        prompt = (
            "Rewrite the latest user message into a standalone retrieval query.\n"
            "The latest user message is authoritative. Conversation memory is only for resolving references.\n"
            "For a follow-up such as 'what about X?', reuse the prior question pattern but replace the prior topic with X.\n"
            "Preserve exact filenames, model names, API names, acronyms, and technical terms.\n"
            "Do not answer the question. Do not explain your reasoning.\n"
            "Return only the standalone retrieval query on one line.\n\n"
            f"Latest user message:\n{question}\n\n"
            f"Conversation memory for reference:\n{memory.as_text() or 'No prior memory.'}\n\n"
            "Standalone retrieval query:"
        )
        complete_prompt = getattr(self.llm, "complete_prompt", None)
        if callable(complete_prompt):
            try:
                rewritten = complete_prompt(
                    prompt,
                    options={
                        "temperature": 0.0,
                        "num_predict": 64,
                        "think": False,
                        "stop": ["\n"],
                    },
                ).strip()
            except Exception as exc:
                return QueryRewriteResult(question, rewrite_used=False, rewrite_error=str(exc))
            if not rewritten:
                return QueryRewriteResult(
                    question,
                    rewrite_used=False,
                    rewrite_error="Query rewrite returned an empty query.",
                )
            return QueryRewriteResult(rewritten, rewrite_used=True)

        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the latest user message into a standalone retrieval query. "
                    "The latest user message is authoritative. Conversation memory is only for resolving references. "
                    "For a follow-up such as 'what about X?', reuse the prior question pattern but replace the prior topic with X. "
                    "Preserve exact filenames, model names, API names, acronyms, and technical terms. "
                    "Do not answer the question. Do not explain your reasoning. "
                    "Return only the standalone retrieval query."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Latest user message:\n{question}\n\n"
                    f"Conversation memory for reference:\n{memory.as_text() or 'No prior memory.'}"
                ),
            },
        ]
        try:
            rewritten = self.llm.complete_chat(
                messages,
                options={"temperature": 0.0, "num_predict": 64, "think": False},
            ).strip()
        except Exception as exc:
            return QueryRewriteResult(question, rewrite_used=False, rewrite_error=str(exc))
        if not rewritten:
            return QueryRewriteResult(
                question,
                rewrite_used=False,
                rewrite_error="Query rewrite returned an empty query.",
            )
        return QueryRewriteResult(rewritten, rewrite_used=True)


def _rewrite_simple_followup(question: str, memory: ConversationMemory) -> str | None:
    match = FOLLOWUP_TOPIC_RE.match(question)
    if match is None:
        return None

    topic = _clean_topic(match.group("topic"))
    if not topic:
        return None

    previous_question = _last_user_question(memory)
    if not previous_question:
        return None

    rewritten, count = TRAILING_TOPIC_RE.subn(
        lambda matched: f"{matched.group('lead')} {topic}{matched.group('punct') or '?'}",
        previous_question.strip(),
        count=1,
    )
    if count:
        return rewritten

    if "ollama" in previous_question.lower() and "endpoint" in previous_question.lower():
        return f"What Ollama endpoint should this app use for {topic}?"

    return f"{previous_question.rstrip('?.!')} for {topic}?"


def _clean_topic(topic: str) -> str:
    return topic.strip().strip("?.!").strip()


def _last_user_question(memory: ConversationMemory) -> str | None:
    for turn in reversed(memory.recent_turns):
        if turn.role.lower() == "user" and turn.content.strip():
            return turn.content.strip()
    return None
