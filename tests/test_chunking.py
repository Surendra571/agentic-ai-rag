"""Tests for document chunking pipeline."""

import pytest

from app.config import get_settings
from app.ingestion.chunker import (
    DocumentChunk,
    DocumentChunker,
    chunk_documents,
)
from app.ingestion.loader import DocumentPage


@pytest.fixture
def sample_document_page() -> DocumentPage:
    content = (
        "Agentic AI systems represent a significant evolution from passive LLM generation. "
        "Instead of producing text in a single feed-forward pass, an agentic system executes "
        "autonomous loops that perceive context, plan actions, and retrieve information.\n\n"
        "Retrieval-Augmented Generation (RAG) grounds language models in external knowledge bases. "
        "In a naive RAG system, documents are fetched and passed to the model unconditionally. "
        "However, in an agentic RAG architecture, retrieval is dynamic and verifiable.\n\n"
        "By evaluating retrieval relevance, detecting potential hallucinations, and rewriting "
        "queries when necessary, agentic RAG provides deterministic reliability for enterprise "
        "production workloads."
    )
    return DocumentPage(text=content, page=12, source="Ebook-Agentic-AI.pdf")


def test_chunks_are_generated(sample_document_page: DocumentPage) -> None:
    chunks = chunk_documents([sample_document_page], chunk_size=200, chunk_overlap=30)
    assert len(chunks) > 1
    for chunk in chunks:
        assert isinstance(chunk, DocumentChunk)
        assert len(chunk.text) > 0


def test_chunk_size_is_respected_approximately(sample_document_page: DocumentPage) -> None:
    chunk_size = 250
    overlap = 30
    chunks = chunk_documents([sample_document_page], chunk_size=chunk_size, chunk_overlap=overlap)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.text) <= chunk_size + 20


def test_chunk_overlap_is_configured() -> None:
    continuous_text = (
        "Sentence one explains agentic AI foundations. "
        "Sentence two discusses how autonomous workflows operate in production. "
        "Sentence three explores vector retrieval mechanics and dense indexing. "
        "Sentence four covers hallucination grading methods and grounding checks. "
        "Sentence five summarizes evaluation frameworks and benchmarks."
    )
    doc_page = DocumentPage(text=continuous_text, page=1, source="test.pdf")
    chunk_size = 130
    chunk_overlap = 45

    chunks = chunk_documents([doc_page], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert len(chunks) >= 3
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i].text
        next_chunk = chunks[i + 1].text

        current_words = current_chunk.split()
        next_words = next_chunk.split()

        shared_words = set(current_words[-5:]) & set(next_words[:8])
        assert len(shared_words) > 0, f"Expected overlapping words between chunk {i} and {i + 1}"


def test_metadata_is_preserved(sample_document_page: DocumentPage) -> None:
    chunks = chunk_documents([sample_document_page], chunk_size=250, chunk_overlap=30)
    assert len(chunks) >= 2
    for i, chunk in enumerate(chunks, start=1):
        assert chunk.source == "Ebook-Agentic-AI.pdf"
        assert chunk.page == 12
        assert chunk.chunk_id == f"page_12_chunk_{i:02d}"
        assert chunk.metadata["source"] == "Ebook-Agentic-AI.pdf"
        assert chunk.metadata["page"] == 12
        assert chunk.metadata["chunk_id"] == chunk.chunk_id


def test_chunk_ids_are_unique() -> None:
    pages = [
        DocumentPage(text="First page content." * 15, page=1, source="Ebook-Agentic-AI.pdf"),
        DocumentPage(text="Second page content." * 15, page=2, source="Ebook-Agentic-AI.pdf"),
        DocumentPage(text="Third page content." * 15, page=3, source="Ebook-Agentic-AI.pdf"),
    ]
    chunks = chunk_documents(pages, chunk_size=200, chunk_overlap=30)
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    assert len(chunk_ids) > 5
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found"


def test_empty_text_is_ignored() -> None:
    empty_pages = [
        DocumentPage(text="   \n\t   ", page=1, source="Ebook-Agentic-AI.pdf"),
        DocumentPage(text="\n\n", page=2, source="Ebook-Agentic-AI.pdf"),
    ]
    chunks = chunk_documents(empty_pages)
    assert chunks == []


def test_chunker_uses_settings_defaults() -> None:
    settings = get_settings()
    chunker = DocumentChunker()
    assert chunker.chunk_size == settings.CHUNK_SIZE
    assert chunker.chunk_overlap == settings.CHUNK_OVERLAP


def test_invalid_overlap_raises_value_error() -> None:
    with pytest.raises(ValueError) as exc_info:
        DocumentChunker(chunk_size=100, chunk_overlap=100)
    assert "strictly less than chunk_size" in str(exc_info.value)
