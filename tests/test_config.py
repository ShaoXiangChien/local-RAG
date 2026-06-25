from pathlib import Path

from rag_demo.config import AppConfig, load_config


def test_default_chat_model_is_non_thinking_for_demo() -> None:
    config = AppConfig()

    assert config.models.llm_model == "llama3.2:3b"
    assert config.generation.think is False


def test_config_file_uses_non_thinking_chat_model() -> None:
    config = load_config(Path("config.toml"))

    assert config.models.llm_model == "llama3.2:3b"
    assert config.models.embedding_model == "embeddinggemma"
    assert config.generation.think is False
