from __future__ import annotations

from collections.abc import Iterator
import re
from typing import Any

from rag_demo.models.base import LLMStreamChunk

NO_THINK_DIRECTIVE = "/no_think"
OLLAMA_STREAM_METRIC_FIELDS = {
    "total_duration": "total_duration_ns",
    "load_duration": "load_duration_ns",
    "prompt_eval_count": "prompt_eval_count",
    "prompt_eval_duration": "prompt_eval_duration_ns",
    "eval_count": "eval_count",
    "eval_duration": "eval_duration_ns",
}


class OllamaLLMClient:
    """Small adapter around the local Ollama Python client."""

    def __init__(self, model: str, host: str = "http://localhost:11434"):
        if "ollama.com" in host:
            raise ValueError("Cloud Ollama hosts are not allowed for this local-only demo.")
        self.model = model
        self.host = host
        self._client = None

    def _ollama_client(self):
        if self._client is None:
            try:
                from ollama import Client
            except ImportError as exc:
                raise RuntimeError("Install the 'ollama' Python package to use Ollama.") from exc
            self._client = Client(host=self.host)
        return self._client

    def healthcheck(self) -> dict[str, Any]:
        try:
            models = self._ollama_client().list()
        except Exception as exc:  # pragma: no cover - depends on local Ollama state
            return {
                "ok": False,
                "provider": "ollama",
                "host": self.host,
                "model": self.model,
                "error": str(exc),
            }
        return {
            "ok": True,
            "provider": "ollama",
            "host": self.host,
            "model": self.model,
            "models": models,
        }

    def complete_chat(
        self, messages: list[dict[str, str]], options: dict[str, Any] | None = None
    ) -> str:
        chat_options, think = _split_chat_options(options)
        response = self._ollama_client().chat(
            model=self.model,
            messages=_prepare_messages(messages, self.model, think),
            options=chat_options,
            think=think,
        )
        return _strip_raw_thinking_content(
            _message_content(response),
            should_strip=_should_strip_raw_thinking(self.model, think),
        )

    def complete_prompt(self, prompt: str, options: dict[str, Any] | None = None) -> str:
        generate_options, think = _split_chat_options(options)
        response = self._ollama_client().generate(
            model=self.model,
            prompt=_prepare_prompt(prompt, self.model, think),
            options=generate_options,
            think=think,
            stream=False,
        )
        return _strip_raw_thinking_content(
            _response_content(response),
            should_strip=_should_strip_raw_thinking(self.model, think),
        )

    def stream_chat(
        self, messages: list[dict[str, str]], options: dict[str, Any] | None = None
    ) -> Iterator[str]:
        for chunk in self.stream_chat_events(messages, options=options):
            if chunk.kind == "content":
                yield chunk.text

    def stream_chat_events(
        self, messages: list[dict[str, str]], options: dict[str, Any] | None = None
    ) -> Iterator[LLMStreamChunk]:
        chat_options, think = _split_chat_options(options)
        content_filter = _RawThinkingContentFilter(
            enabled=_should_strip_raw_thinking(self.model, think)
        )
        stream = self._ollama_client().chat(
            model=self.model,
            messages=_prepare_messages(messages, self.model, think),
            options=chat_options,
            think=think,
            stream=True,
        )
        for chunk in stream:
            thinking = _message_thinking(chunk)
            if thinking:
                yield LLMStreamChunk(kind="thinking", text=thinking)
            content = _message_content(chunk)
            if content:
                for visible in content_filter.feed(content):
                    yield LLMStreamChunk(kind="content", text=visible)
            metrics = _stream_metrics(chunk)
            if metrics:
                yield LLMStreamChunk(kind="metrics", text="", metadata=metrics)
        trailing = content_filter.flush()
        if trailing:
            yield LLMStreamChunk(kind="content", text=trailing)


def _split_chat_options(options: dict[str, Any] | None) -> tuple[dict[str, Any], Any]:
    chat_options = dict(options or {})
    think = chat_options.pop("think", False)
    return chat_options, think


def _prepare_messages(
    messages: list[dict[str, str]],
    model: str,
    think: Any,
) -> list[dict[str, str]]:
    prepared = [dict(message) for message in messages]
    if think is not False or "qwen" not in model.lower():
        return prepared

    if prepared and prepared[0].get("role") == "system":
        system_content = str(prepared[0].get("content") or "")
        if NO_THINK_DIRECTIVE not in system_content:
            prepared[0]["content"] = f"{NO_THINK_DIRECTIVE}\n{system_content}"

    target_index = _last_user_message_index(prepared)
    if target_index is None:
        return prepared

    content = str(prepared[target_index].get("content") or "")
    if NO_THINK_DIRECTIVE in content:
        return prepared
    prepared[target_index]["content"] = f"{NO_THINK_DIRECTIVE}\n{content}"
    return prepared


def _prepare_prompt(prompt: str, model: str, think: Any) -> str:
    if think is not False or "qwen" not in model.lower() or NO_THINK_DIRECTIVE in prompt:
        return prompt
    return f"{NO_THINK_DIRECTIVE}\n{prompt}"


def _last_user_message_index(messages: list[dict[str, str]]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return len(messages) - 1 if messages else None


def _message_content(response: Any) -> str:
    if isinstance(response, dict):
        message = response.get("message") or {}
        if isinstance(message, dict):
            return str(message.get("content") or "")
    message = getattr(response, "message", None)
    if message is not None:
        return str(getattr(message, "content", "") or "")
    return ""


def _response_content(response: Any) -> str:
    if isinstance(response, dict):
        return str(response.get("response") or "")
    return str(getattr(response, "response", "") or "")


def _message_thinking(response: Any) -> str:
    if isinstance(response, dict):
        message = response.get("message") or {}
        if isinstance(message, dict):
            return str(message.get("thinking") or "")
    message = getattr(response, "message", None)
    if message is not None:
        return str(getattr(message, "thinking", "") or "")
    return ""


def _stream_metrics(response: Any) -> dict[str, int]:
    metrics: dict[str, int] = {}
    for source_key, target_key in OLLAMA_STREAM_METRIC_FIELDS.items():
        value = _response_value(response, source_key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            metrics[target_key] = int(value)
    return metrics


def _response_value(response: Any, key: str) -> Any:
    if isinstance(response, dict):
        return response.get(key)
    return getattr(response, key, None)


def _should_strip_raw_thinking(model: str, think: Any) -> bool:
    return think is False and "qwen" in model.lower()


def _strip_raw_thinking_content(text: str, should_strip: bool) -> str:
    if not should_strip or not text:
        return text
    marker = "</think>"
    if marker in text:
        return text.split(marker, 1)[1].lstrip()
    if "<think>" in text:
        return ""
    if not _looks_like_reasoning_preamble(text):
        return text
    return _extract_answer_from_cutoff_thinking(text)


class _RawThinkingContentFilter:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._buffer = ""
        self._released = False

    def feed(self, text: str) -> list[str]:
        if not self.enabled or self._released:
            return [text]

        self._buffer += text
        marker = "</think>"
        if marker not in self._buffer:
            return []

        _, visible = self._buffer.split(marker, 1)
        self._buffer = ""
        self._released = True
        visible = visible.lstrip()
        return [visible] if visible else []

    def flush(self) -> str:
        if not self.enabled or self._released:
            return ""
        visible = _strip_raw_thinking_content(self._buffer, should_strip=True)
        self._buffer = ""
        return visible


def _extract_answer_from_cutoff_thinking(text: str) -> str:
    for pattern in (
        r"(?is)\bso\s+the\s+answer\s+is\s*:?\s*(?P<answer>.+)$",
        r"(?is)\bthe\s+answer\s+is\s*:?\s*(?P<answer>.+)$",
        r"(?is)\bfinal\s+answer\s*:?\s*(?P<answer>.+)$",
    ):
        match = re.search(pattern, text)
        if match:
            return _trim_extracted_answer(match.group("answer"))
    return ""


def _looks_like_reasoning_preamble(text: str) -> bool:
    normalized = text.lstrip().lower()
    return normalized.startswith(
        (
            "looking at ",
            "looking through ",
            "we are given ",
            "we need ",
            "i need ",
            "let me ",
            "from the context",
            "from the retrieved context",
            "the user is asking",
        )
    )


def _trim_extracted_answer(answer: str) -> str:
    trimmed = answer.strip()
    for separator in (
        "\n\n",
        "\nBut ",
        "\nHowever",
        "\nI need",
        "\nWe ",
        "\nLet ",
    ):
        if separator in trimmed:
            trimmed = trimmed.split(separator, 1)[0].strip()
    return trimmed.strip("` ")
