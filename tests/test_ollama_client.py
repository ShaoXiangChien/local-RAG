from rag_demo.models.ollama_client import OllamaLLMClient


class FakeOllamaClient:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(
                [
                    {"message": {"content": "Local "}},
                    {"message": {"content": "answer"}},
                ]
            )
        return {"message": {"content": "complete"}}

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return {"response": "generated"}


def test_ollama_client_disables_thinking_by_default_for_non_streaming_chat() -> None:
    fake = FakeOllamaClient()
    client = OllamaLLMClient("qwen3:4b")
    client._client = fake

    answer = client.complete_chat([{"role": "user", "content": "hello"}])

    assert answer == "complete"
    assert fake.calls[0]["think"] is False
    assert fake.calls[0]["options"] == {}


def test_ollama_client_can_generate_prompt_completion_with_thinking_disabled() -> None:
    fake = FakeOllamaClient()
    client = OllamaLLMClient("qwen3:4b")
    client._client = fake

    answer = client.complete_prompt("Return only a query.", options={"stop": ["\n"]})

    assert answer == "generated"
    assert fake.calls[0]["prompt"].startswith("/no_think\n")
    assert fake.calls[0]["think"] is False
    assert fake.calls[0]["options"] == {"stop": ["\n"]}


def test_ollama_client_extracts_think_from_options_for_streaming_chat() -> None:
    fake = FakeOllamaClient()
    client = OllamaLLMClient("qwen3:4b")
    client._client = fake

    answer = "".join(
        client.stream_chat(
            [{"role": "user", "content": "hello"}],
            options={"temperature": 0.2, "think": "low"},
        )
    )

    assert answer == "Local answer"
    assert fake.calls[0]["think"] == "low"
    assert fake.calls[0]["options"] == {"temperature": 0.2}


def test_ollama_client_streams_thinking_separately_from_content() -> None:
    class ThinkingFakeOllamaClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return iter(
                [
                    {"message": {"thinking": "check sources"}},
                    {"message": {"content": "Use POST /api/chat."}},
                ]
            )

    fake = ThinkingFakeOllamaClient()
    client = OllamaLLMClient("qwen3:4b")
    client._client = fake

    chunks = list(
        client.stream_chat_events(
            [{"role": "user", "content": "what about chat?"}],
            options={"think": True},
        )
    )

    assert [(chunk.kind, chunk.text) for chunk in chunks] == [
        ("thinking", "check sources"),
        ("content", "Use POST /api/chat."),
    ]


def test_ollama_client_suppresses_qwen_raw_thinking_content_when_thinking_is_disabled() -> None:
    class RawThinkingFakeOllamaClient:
        def chat(self, **kwargs):
            return iter(
                [
                    {"message": {"content": "Looking at the retrieved context, "}},
                    {"message": {"content": "I can see the endpoint.\n</think>\n\n"}},
                    {"message": {"content": "POST /api/embed"}},
                ]
            )

    client = OllamaLLMClient("qwen3:4b")
    client._client = RawThinkingFakeOllamaClient()

    chunks = list(
        client.stream_chat_events(
            [{"role": "user", "content": "What endpoint should embeddings use?"}],
            options={"think": False},
        )
    )

    assert [(chunk.kind, chunk.text) for chunk in chunks] == [
        ("content", "POST /api/embed")
    ]


def test_ollama_client_extracts_answer_if_qwen_thinking_is_cut_before_close_tag() -> None:
    class CutoffThinkingFakeOllamaClient:
        def chat(self, **kwargs):
            return iter(
                [
                    {
                        "message": {
                            "content": (
                                "Looking at the retrieved context, I can see the endpoint.\n\n"
                                "So the answer is: POST /api/embed\n\n"
                                "But note that I should cite chunks"
                            )
                        }
                    }
                ]
            )

    client = OllamaLLMClient("qwen3:4b")
    client._client = CutoffThinkingFakeOllamaClient()

    answer = "".join(
        client.stream_chat(
            [{"role": "user", "content": "What endpoint should embeddings use?"}],
            options={"think": False},
        )
    )

    assert answer == "POST /api/embed"


def test_ollama_client_adds_qwen_no_think_directive_without_mutating_messages() -> None:
    fake = FakeOllamaClient()
    client = OllamaLLMClient("qwen3:4b")
    client._client = fake
    messages = [
        {"role": "system", "content": "Answer directly."},
        {"role": "user", "content": "What endpoint should embeddings use?"},
    ]

    client.complete_chat(messages)

    sent_messages = fake.calls[0]["messages"]
    assert sent_messages[0]["content"] == "/no_think\nAnswer directly."
    assert sent_messages[1]["content"].startswith("/no_think\n")
    assert messages[1]["content"] == "What endpoint should embeddings use?"


def test_ollama_client_emits_final_stream_metrics() -> None:
    class MetricsFakeOllamaClient:
        def chat(self, **kwargs):
            return iter(
                [
                    {"message": {"content": "Fast answer"}},
                    {
                        "done": True,
                        "total_duration": 2_500_000_000,
                        "load_duration": 100_000_000,
                        "prompt_eval_count": 120,
                        "prompt_eval_duration": 600_000_000,
                        "eval_count": 25,
                        "eval_duration": 1_250_000_000,
                    },
                ]
            )

    client = OllamaLLMClient("llama3.2:3b")
    client._client = MetricsFakeOllamaClient()

    chunks = list(
        client.stream_chat_events(
            [{"role": "user", "content": "Explain local inference metrics."}],
            options={"think": False},
        )
    )

    assert [(chunk.kind, chunk.text) for chunk in chunks] == [
        ("content", "Fast answer"),
        ("metrics", ""),
    ]
    assert chunks[-1].metadata == {
        "total_duration_ns": 2_500_000_000,
        "load_duration_ns": 100_000_000,
        "prompt_eval_count": 120,
        "prompt_eval_duration_ns": 600_000_000,
        "eval_count": 25,
        "eval_duration_ns": 1_250_000_000,
    }
