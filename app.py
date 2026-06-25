from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from rag_demo.chat.memory import ConversationMemoryBuffer
from rag_demo.chat.prompt import PromptBuilder
from rag_demo.chat.query_rewrite import QueryRewriteService
from rag_demo.chat.service import ChatService
from rag_demo.chat.summarizer import ConversationSummarizer
from rag_demo.chat.token_budget import ContextBudgetConfig, HeuristicTokenCounter, TokenBudgeter
from rag_demo.config import AppConfig, ensure_local_paths, load_config
from rag_demo.embeddings import OllamaEmbedder
from rag_demo.ingestion import ChunkingConfig, DocumentIngestionPipeline, RecursiveChunker
from rag_demo.models import OllamaLLMClient
from rag_demo.retrieval import HybridRetriever, LanceDBVectorStore
from rag_demo.storage import ConversationStore, DocumentManifest
from rag_demo.ui.chat_page import render_chat_page
from rag_demo.ui.documents_page import render_documents_page
from rag_demo.ui.sidebar import render_sidebar


@dataclass
class Services:
    config: AppConfig
    llm: OllamaLLMClient
    manifest: DocumentManifest
    memory_buffer: ConversationMemoryBuffer
    ingestion_pipeline: DocumentIngestionPipeline
    chat_service: ChatService


@st.cache_resource
def build_services() -> Services:
    config = load_config()
    ensure_local_paths(config)

    llm = OllamaLLMClient(
        model=config.models.llm_model,
        host=config.models.ollama_host,
    )
    embedder = OllamaEmbedder(
        model_name=config.models.embedding_model,
        host=config.models.ollama_host,
    )
    manifest = DocumentManifest(config.paths.manifest_db)
    conversation_store = ConversationStore(config.paths.manifest_db)
    memory_buffer = ConversationMemoryBuffer(
        conversation_store,
        recent_turns=config.memory.recent_turns,
    )
    vector_store = LanceDBVectorStore(
        config.paths.index_dir,
        full_text_index=config.retrieval.full_text_index,
    )
    chunker = RecursiveChunker(
        ChunkingConfig(
            chunk_token_size=config.retrieval.chunk_token_size,
            chunk_token_overlap=config.retrieval.chunk_token_overlap,
        )
    )
    ingestion_pipeline = DocumentIngestionPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        manifest=manifest,
        upload_dir=config.paths.upload_dir,
        embedding_model=config.models.embedding_model,
    )
    retriever = HybridRetriever(
        embedder=embedder,
        store=vector_store,
        mode=config.retrieval.mode,
        top_k=config.retrieval.top_k,
    )
    budget_config = ContextBudgetConfig(**config.context_budget.__dict__)
    prompt_builder = PromptBuilder(
        TokenBudgeter(
            HeuristicTokenCounter(),
            budget_config,
        )
    )
    query_rewriter = QueryRewriteService(llm, enabled=config.query_rewrite.enabled)
    summarizer = ConversationSummarizer(
        llm,
        summary_token_budget=config.memory.summary_token_budget,
        enabled=config.memory.enabled and config.memory.update_after_each_answer,
    )
    chat_service = ChatService(
        llm=llm,
        retriever=retriever,
        query_rewriter=query_rewriter,
        prompt_builder=prompt_builder,
        memory_buffer=memory_buffer,
        summarizer=summarizer,
        top_k=config.retrieval.top_k,
        generation_options={
            "num_ctx": config.generation.num_ctx,
            "num_predict": config.generation.num_predict,
            "temperature": config.generation.temperature,
            "top_p": config.generation.top_p,
            "think": config.generation.think,
        },
    )
    return Services(
        config=config,
        llm=llm,
        manifest=manifest,
        memory_buffer=memory_buffer,
        ingestion_pipeline=ingestion_pipeline,
        chat_service=chat_service,
    )


def main() -> None:
    st.set_page_config(page_title="Edge AI Local RAG", page_icon=":material/hub:", layout="wide")
    services = build_services()
    page = render_sidebar(services.config, services.llm, services.manifest)
    if page == "Documents":
        render_documents_page(
            services.ingestion_pipeline,
            services.manifest,
            services.config,
        )
    else:
        render_chat_page(
            services.chat_service,
            services.manifest,
            services.llm,
            services.memory_buffer,
        )


if __name__ == "__main__":
    main()
