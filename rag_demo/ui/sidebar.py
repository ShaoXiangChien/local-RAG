from __future__ import annotations

import streamlit as st


def render_sidebar(config, llm, manifest) -> str:
    st.sidebar.title("Edge AI RAG")
    page = st.sidebar.radio("Page", ["Chat", "Documents"], label_visibility="collapsed")

    st.sidebar.divider()
    health = llm.healthcheck()
    if health.get("ok"):
        st.sidebar.success("Ollama reachable")
    else:
        st.sidebar.error("Ollama unavailable")
        st.sidebar.caption(str(health.get("error", "Local model server not reachable.")))

    st.sidebar.caption(f"LLM: `{config.models.llm_model}`")
    st.sidebar.caption(f"Embedding: `{config.models.embedding_model}`")
    st.sidebar.caption(f"Retrieval: `{config.retrieval.mode}`")
    st.sidebar.caption(
        f"Rewrite: `{'on' if config.query_rewrite.enabled else 'off'}` | "
        f"Memory: `{'on' if config.memory.enabled else 'off'}`"
    )

    indexed_count = len(manifest.list_documents())
    st.sidebar.caption(f"Indexed docs: `{indexed_count}`")

    return page
