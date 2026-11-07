"""Streamlit interface for the RAG document assistant."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Protocol

import streamlit as st
from dotenv import load_dotenv

from src.chat_manager import export_chat_json, export_chat_text, new_message
from src.document_loader import DocumentProcessingError, load_document
from src.embeddings import DEFAULT_EMBEDDING_MODEL, create_embeddings
from src.rag_pipeline import (
    GeminiRequestError,
    MissingAPIKeyError,
    RAGPipeline,
)
from src.text_splitter import split_documents
from src.vector_store import VectorStore
from utils.helpers import MAX_UPLOAD_SIZE_BYTES, validate_upload

load_dotenv()

st.set_page_config(
    page_title="Document Assistant",
    page_icon="📚",
    layout="wide",
)


class _UploadedFile(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


def _initialize_state() -> None:
    defaults = {
        "source_documents": {},
        "document_status": {},
        "vector_store": VectorStore(),
        "indexed_chunks": 0,
        "messages": [],
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _rebuild_index() -> None:
    documents = [
        document
        for collection in st.session_state.source_documents.values()
        for document in collection
    ]
    chunks = split_documents(documents)
    if not chunks:
        st.session_state.vector_store = VectorStore()
        st.session_state.indexed_chunks = 0
        return
    with st.spinner("Creating embeddings and rebuilding the search index..."):
        embeddings = create_embeddings(st.session_state.embedding_model)
        vector_store = VectorStore()
        vector_store.build(chunks, embeddings)
        st.session_state.vector_store = vector_store
        st.session_state.indexed_chunks = len(chunks)


def _process_uploads(files: list[_UploadedFile]) -> None:
    added_any = False
    added_keys: list[str] = []
    for uploaded_file in files:
        content = uploaded_file.getvalue()
        error = validate_upload(uploaded_file.name, len(content))
        fingerprint = hashlib.sha256(content).hexdigest()
        key = f"{uploaded_file.name}:{fingerprint}"
        if error:
            st.session_state.document_status[key] = {
                "name": uploaded_file.name,
                "status": "Error",
                "detail": error,
            }
            continue
        if key in st.session_state.source_documents:
            continue
        try:
            loaded = load_document(uploaded_file.name, content)
            st.session_state.source_documents[key] = loaded
            st.session_state.document_status[key] = {
                "name": uploaded_file.name,
                "status": "Processed",
                "detail": f"{len(loaded)} page/document section(s) extracted",
            }
            added_any = True
            added_keys.append(key)
        except DocumentProcessingError as exc:
            st.session_state.document_status[key] = {
                "name": uploaded_file.name,
                "status": "Error",
                "detail": str(exc),
            }

    if added_any:
        try:
            _rebuild_index()
            st.success("Documents processed and the search index was rebuilt.")
        except Exception as exc:
            for key in added_keys:
                st.session_state.source_documents.pop(key, None)
                st.session_state.document_status[key] = {
                    "name": key.split(":", 1)[0],
                    "status": "Error",
                    "detail": f"Could not add document to the search index: {exc}",
                }
            st.error(f"Could not build the document index: {exc}")


def _show_sources(message: dict[str, object]) -> None:
    sources = message.get("sources", [])
    if sources:
        with st.expander(f"Inspect retrieved sources ({len(sources)})"):
            for source in sources:
                source_data = dict(source)
                location = source_data.get("filename", "Unknown source")
                if source_data.get("page") is not None:
                    location += f" — page {source_data['page']}"
                st.markdown(f"**{location}**")
                st.caption(
                    f"Retrieved passage · distance {source_data['distance']:.3f}"
                )
                st.text(source_data.get("passage", ""))


def _show_message(message: dict[str, object]) -> None:
    with st.chat_message(str(message.get("role", "assistant"))):
        st.markdown(str(message.get("content", "")))
        _show_sources(message)


def main() -> None:
    _initialize_state()
    st.title("📚 Intelligent Document Assistant")
    st.caption("Ask questions grounded in your uploaded PDF, TXT, and DOCX documents.")

    with st.sidebar:
        st.header("Your documents")
        uploaded_files = st.file_uploader(
            "Upload one or more documents",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            help=f"Maximum {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB per file.",
        )
        if st.button("Process documents", type="primary", disabled=not uploaded_files):
            _process_uploads(uploaded_files)

        st.divider()
        st.subheader("Processing status")
        if not st.session_state.document_status:
            st.caption("No documents uploaded yet.")
        for status in st.session_state.document_status.values():
            icon = "✅" if status["status"] == "Processed" else "❌"
            st.markdown(f"{icon} **{status['name']}** — {status['status']}")
            st.caption(status["detail"])

        indexed_documents = list(st.session_state.source_documents)
        if indexed_documents:
            selected_document = st.selectbox(
                "Remove a processed document",
                options=indexed_documents,
                format_func=lambda key: Path(key.split(":", 1)[0]).name,
            )
            if st.button("Remove selected document"):
                removed_documents = st.session_state.source_documents.pop(
                    selected_document
                )
                removed_status = st.session_state.document_status.pop(
                    selected_document, None
                )
                try:
                    _rebuild_index()
                    st.success("Document removed and the search index was rebuilt.")
                except Exception as exc:
                    st.session_state.source_documents[selected_document] = removed_documents
                    if removed_status is not None:
                        st.session_state.document_status[selected_document] = removed_status
                    st.error(f"Could not rebuild the document index: {exc}")

        st.divider()
        st.subheader("Retrieval settings")
        top_k = st.slider("Top matching passages", min_value=1, max_value=10, value=4)
        max_distance = st.slider(
            "Maximum L2 distance",
            min_value=0.1,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Smaller values require closer semantic matches.",
        )
        embedding_model = st.text_input(
            "Embedding model",
            value=st.session_state.embedding_model,
            help="Hugging Face Sentence Transformers model name.",
        )
        if embedding_model.strip() and embedding_model != st.session_state.embedding_model:
            previous_model = st.session_state.embedding_model
            st.session_state.embedding_model = embedding_model.strip()
            if st.session_state.source_documents:
                try:
                    _rebuild_index()
                except Exception as exc:
                    st.session_state.embedding_model = previous_model
                    st.error(f"Could not rebuild using this embedding model: {exc}")

        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        st.caption(f"Gemini model: `{model_name}`")
        st.metric("Indexed chunks", st.session_state.indexed_chunks)

        if st.session_state.source_documents and st.button("Rebuild vector database"):
            try:
                _rebuild_index()
                st.success("Vector database rebuilt.")
            except Exception as exc:
                st.error(f"Could not rebuild the vector database: {exc}")

        if st.button("Clear documents", disabled=not st.session_state.document_status):
            st.session_state.source_documents.clear()
            st.session_state.document_status.clear()
            st.session_state.vector_store = VectorStore()
            st.session_state.indexed_chunks = 0
            st.session_state.messages.clear()
            st.rerun()

    st.subheader("Chat")
    if not st.session_state.messages:
        st.info(
            "Welcome! Upload and process documents in the sidebar, then ask a "
            "question. Each question is answered using retrieved document passages."
        )

    for existing_message in st.session_state.messages:
        _show_message(existing_message)

    question = st.chat_input(
        "Ask a question about your documents",
        disabled=st.session_state.indexed_chunks == 0,
    )
    if question:
        st.session_state.messages.append(new_message("user", question))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching your documents and preparing an answer..."):
                try:
                    embeddings = create_embeddings(st.session_state.embedding_model)
                    pipeline = RAGPipeline(
                        vector_store=st.session_state.vector_store,
                        embeddings=embeddings,
                        model_name=model_name,
                    )
                    result = pipeline.answer(
                        question,
                        top_k=top_k,
                        max_distance=max_distance,
                    )
                    message = new_message(
                        "assistant",
                        result.answer,
                        sources=[
                            {
                                "filename": source.filename,
                                "page": source.page,
                                "passage": source.passage,
                                "distance": source.distance,
                            }
                            for source in result.sources
                        ],
                    )
                    st.session_state.messages.append(message)
                    st.markdown(result.answer)
                    _show_sources(message)
                except MissingAPIKeyError as exc:
                    st.error(str(exc))
                    st.session_state.messages.append(
                        new_message("assistant", f"Configuration error: {exc}")
                    )
                except (GeminiRequestError, ValueError) as exc:
                    st.error(str(exc))
                    st.session_state.messages.append(
                        new_message("assistant", f"Unable to answer: {exc}")
                    )
                except Exception as exc:
                    st.error(f"An unexpected error occurred: {exc}")
                    st.session_state.messages.append(
                        new_message("assistant", f"An unexpected error occurred: {exc}")
                    )

    if st.session_state.messages:
        st.divider()
        left, right = st.columns(2)
        with left:
            st.download_button(
                "Download chat as TXT",
                data=export_chat_text(st.session_state.messages),
                file_name="document-assistant-chat.txt",
                mime="text/plain",
            )
        with right:
            st.download_button(
                "Download chat as JSON",
                data=export_chat_json(st.session_state.messages),
                file_name="document-assistant-chat.json",
                mime="application/json",
            )
        if st.button("Reset conversation"):
            st.session_state.messages.clear()
            st.rerun()


if __name__ == "__main__":
    main()
