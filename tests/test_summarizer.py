from rag_demo.chat.summarizer import ConversationSummarizer
from rag_demo.models.base import LocalLLMClient


class FakeSummaryLLM(LocalLLMClient):
    def __init__(self, response: str):
        self.response = response
        self.options = None

    def healthcheck(self):
        return {"ok": True}

    def complete_chat(self, messages, options=None):
        self.options = options
        return self.response

    def stream_chat(self, messages, options=None):
        yield self.complete_chat(messages, options)


def test_summarizer_caps_helper_generation() -> None:
    llm = FakeSummaryLLM("A compact summary.")
    summarizer = ConversationSummarizer(llm, summary_token_budget=600)

    result = summarizer.update_summary("", "What endpoint?", "Use /api/embed.")

    assert result.summary == "A compact summary."
    assert llm.options == {"temperature": 0.0, "num_predict": 160}
