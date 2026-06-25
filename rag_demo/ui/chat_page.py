from __future__ import annotations

from uuid import uuid4

import streamlit as st

PIPELINE_STEPS = [
    "memory",
    "query_rewrite",
    "retrieval",
    "prompt",
    "generation",
    "memory_update",
    "summary",
]
PIPELINE_STEP_LABELS = {
    "memory": "Load memory",
    "query_rewrite": "Rewrite query",
    "retrieval": "Retrieve sources",
    "prompt": "Build prompt",
    "generation": "Generate answer",
    "memory_update": "Save turn",
    "summary": "Summarize memory",
}


def render_chat_page(chat_service, manifest, llm, memory_buffer) -> None:
    st.header("Chat")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    left, right = st.columns(2)
    with left:
        if st.button("Clear chat"):
            st.session_state.chat_messages = []
            st.rerun()
    with right:
        if st.button("Reset memory"):
            memory_buffer.reset(st.session_state.session_id)
            st.success("Conversation memory reset.")

    health = llm.healthcheck()
    if not health.get("ok"):
        st.warning(
            "Local model server is not reachable at the configured Ollama host. "
            "Install/start Ollama and pull the configured models before chatting."
        )
    if not manifest.has_indexed_documents():
        st.info("Upload and index documents before asking grounded questions.")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask a question about your indexed documents")
    if not question:
        return

    st.session_state.chat_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if not manifest.has_indexed_documents():
            answer = "No documents are indexed yet."
            st.write(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            return

        pipeline_events = []
        thinking_parts = []
        status = st.status("Running local RAG pipeline...", expanded=True)
        with status:
            progress = st.progress(0)
            trace_placeholder = st.empty()
        thinking_container = st.empty()

        def on_pipeline_event(event) -> None:
            pipeline_events.append(event)
            completed = _completed_step_count(pipeline_events)
            progress.progress(min(completed / len(PIPELINE_STEPS), 1.0))
            trace_placeholder.table(_pipeline_rows(pipeline_events))
            if event.status == "failed":
                status.update(label="Local RAG pipeline failed", state="error", expanded=True)
            elif completed == len(PIPELINE_STEPS):
                status.update(label="Local RAG pipeline complete", state="complete", expanded=False)
            else:
                status.update(label=f"Running: {_step_label(event.step)}", state="running")

        def on_model_thinking(delta: str) -> None:
            thinking_parts.append(delta)
            with thinking_container.container():
                with st.expander("Model thinking", expanded=False):
                    st.markdown("".join(thinking_parts))

        try:
            result = chat_service.ask(
                st.session_state.session_id,
                question,
                on_event=on_pipeline_event,
                on_thinking=on_model_thinking,
            )
            answer = st.write_stream(result.stream)
        except Exception as exc:
            answer = f"Local RAG pipeline failed: {exc}"
            status.update(label="Local RAG pipeline failed", state="error", expanded=True)
            st.error(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            return

        status.update(label="Local RAG pipeline complete", state="complete", expanded=False)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        _render_pipeline_trace(pipeline_events)
        _render_debug(result)
        _render_sources(result.sources)


def _step_label(step: str) -> str:
    return PIPELINE_STEP_LABELS.get(step, step.replace("_", " ").title())


def _completed_step_count(events) -> int:
    completed = {
        event.step
        for event in events
        if event.step in PIPELINE_STEPS and event.status in {"completed", "failed"}
    }
    return len(completed)


def _pipeline_rows(events) -> list[dict[str, str]]:
    latest_by_step = {}
    for event in events:
        latest_by_step[event.step] = event

    rows = []
    for step in PIPELINE_STEPS:
        event = latest_by_step.get(step)
        if event is None:
            rows.append({"Step": _step_label(step), "Status": "pending", "Time": "", "Detail": ""})
            continue
        rows.append(
            {
                "Step": _step_label(step),
                "Status": event.status,
                "Time": f"{event.elapsed_ms} ms" if event.elapsed_ms is not None else "",
                "Detail": event.detail or event.message,
            }
        )
    return rows


def _render_pipeline_trace(events) -> None:
    with st.expander("Pipeline trace", expanded=False):
        st.table(_pipeline_rows(events))


def _render_debug(result) -> None:
    with st.expander("Retrieval debug", expanded=False):
        st.write(f"Rewritten query: `{result.rewrite.rewritten_query}`")
        st.write(f"Rewrite used: `{result.rewrite.rewrite_used}`")
        if result.rewrite.rewrite_error:
            st.warning(result.rewrite.rewrite_error)
        st.write(f"Prompt tokens estimated: `{result.prompt.total_prompt_tokens}`")
        st.write(f"Dropped sources: `{result.prompt.dropped_source_count}`")


def _render_sources(sources) -> None:
    if not sources:
        st.warning("No source chunks were included in the prompt.")
        return
    st.subheader("Sources")
    for source in sources:
        with st.expander(
            f"{source.filename} | chunk {source.chunk_index} | score {source.score:.3f}",
            expanded=False,
        ):
            st.caption(f"chunk_id: `{source.chunk_id}`")
            st.write(source.excerpt)
            st.code(source.text)
