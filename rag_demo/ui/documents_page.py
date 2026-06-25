from __future__ import annotations

import streamlit as st


def render_documents_page(pipeline, manifest, config) -> None:
    st.header("Documents")

    if manifest.embedding_model_mismatch(config.models.embedding_model):
        st.warning(
            "Some indexed documents were embedded with a different model. "
            "Delete and re-index them before relying on retrieval quality."
        )

    uploads = st.file_uploader(
        "Upload Markdown or text files",
        type=["md", "txt"],
        accept_multiple_files=True,
    )
    if st.button("Index uploads", disabled=not uploads, type="primary"):
        for upload in uploads or []:
            with st.spinner(f"Indexing {upload.name}"):
                try:
                    result = pipeline.ingest_upload(upload.name, upload.getvalue())
                except Exception as exc:
                    st.error(f"{upload.name}: {exc}")
                else:
                    if result.status == "skipped":
                        st.info(f"{result.filename}: {result.message}")
                    else:
                        st.success(f"{result.filename}: {result.message}")

    st.divider()
    documents = manifest.list_documents()
    if not documents:
        st.info("No indexed documents yet.")
        return

    rows = [
        {
            "filename": doc.filename,
            "type": doc.file_type,
            "size_bytes": doc.size_bytes,
            "chunks": doc.chunk_count,
            "embedding_model": doc.embedding_model,
            "status": doc.status,
            "updated_at": doc.updated_at.strftime("%Y-%m-%d %H:%M"),
            "doc_id": doc.doc_id,
        }
        for doc in documents
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)

    selected = st.selectbox(
        "Document actions",
        options=documents,
        format_func=lambda doc: f"{doc.filename} ({doc.chunk_count} chunks)",
    )
    if st.button("Delete selected document", type="secondary"):
        pipeline.delete_document(selected.doc_id)
        st.success(f"Deleted {selected.filename}")
        st.rerun()
