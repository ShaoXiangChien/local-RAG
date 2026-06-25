from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib


@dataclass(frozen=True)
class PathConfig:
    upload_dir: Path = Path("data/uploads")
    index_dir: Path = Path("data/index")
    manifest_db: Path = Path("data/app.db")


@dataclass(frozen=True)
class ModelConfig:
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2:3b"
    embedding_provider: str = "ollama"
    embedding_model: str = "embeddinggemma"
    ollama_host: str = "http://localhost:11434"


@dataclass(frozen=True)
class GenerationConfig:
    num_ctx: int = 6144
    num_predict: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    think: bool | str | None = False


@dataclass(frozen=True)
class RetrievalConfig:
    vector_store: str = "lancedb"
    mode: str = "hybrid"
    top_k: int = 4
    chunk_token_size: int = 450
    chunk_token_overlap: int = 70
    full_text_index: bool = True


@dataclass(frozen=True)
class QueryRewriteConfig:
    enabled: bool = True
    fallback_to_original_query: bool = True


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool = True
    summary_token_budget: int = 600
    recent_turns: int = 2
    update_after_each_answer: bool = True


@dataclass(frozen=True)
class ContextBudgetSettings:
    max_working_context_tokens: int = 6000
    reserved_response_tokens: int = 800
    reserved_system_tokens: int = 300
    reserved_memory_summary_tokens: int = 600
    reserved_recent_turn_tokens: int = 500
    reserved_query_tokens: int = 350
    safety_margin_tokens: int = 50


@dataclass(frozen=True)
class AppConfig:
    paths: PathConfig = PathConfig()
    models: ModelConfig = ModelConfig()
    generation: GenerationConfig = GenerationConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    query_rewrite: QueryRewriteConfig = QueryRewriteConfig()
    memory: MemoryConfig = MemoryConfig()
    context_budget: ContextBudgetSettings = ContextBudgetSettings()


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    return value if isinstance(value, dict) else {}


def load_config(path: str | Path = "config.toml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    paths = _section(raw, "paths")
    models = _section(raw, "models")
    generation = _section(raw, "generation")
    retrieval = _section(raw, "retrieval")
    query_rewrite = _section(raw, "query_rewrite")
    memory = _section(raw, "memory")
    context_budget = _section(raw, "context_budget")

    return AppConfig(
        paths=PathConfig(
            upload_dir=Path(paths.get("upload_dir", PathConfig.upload_dir)),
            index_dir=Path(paths.get("index_dir", PathConfig.index_dir)),
            manifest_db=Path(paths.get("manifest_db", PathConfig.manifest_db)),
        ),
        models=ModelConfig(**{**ModelConfig().__dict__, **models}),
        generation=GenerationConfig(**{**GenerationConfig().__dict__, **generation}),
        retrieval=RetrievalConfig(**{**RetrievalConfig().__dict__, **retrieval}),
        query_rewrite=QueryRewriteConfig(
            **{**QueryRewriteConfig().__dict__, **query_rewrite}
        ),
        memory=MemoryConfig(**{**MemoryConfig().__dict__, **memory}),
        context_budget=ContextBudgetSettings(
            **{**ContextBudgetSettings().__dict__, **context_budget}
        ),
    )


def ensure_local_paths(config: AppConfig) -> None:
    config.paths.upload_dir.mkdir(parents=True, exist_ok=True)
    config.paths.index_dir.mkdir(parents=True, exist_ok=True)
    config.paths.manifest_db.parent.mkdir(parents=True, exist_ok=True)
