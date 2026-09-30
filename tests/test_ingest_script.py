"""Unit tests for the scripts/ingest.py command."""

from io import StringIO
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import pytest

from app.ingestion.chunker import DocumentChunk
from app.ingestion.embedder import EmbeddedChunk
from app.ingestion.loader import DocumentPage, EmptyPDFError, PDFNotFoundError
from app.retrieval.vector_store import VectorStoreError
from scripts.ingest import IngestionSummary, main, run_ingestion


@pytest.fixture
def sample_pages() -> list[DocumentPage]:
    """Sample extracted document pages."""
    return [
        DocumentPage(
            page=1,
            text="Autonomous loops perceive environment and execute tools.",
            source="Ebook-Agentic-AI.pdf",
        ),
        DocumentPage(
            page=2,
            text="Agentic RAG evaluates relevance before generation.",
            source="Ebook-Agentic-AI.pdf",
        ),
    ]


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    """Sample document chunks with deterministic IDs."""
    return [
        DocumentChunk(
            chunk_id="page_01_chunk_01",
            source="Ebook-Agentic-AI.pdf",
            page=1,
            text="Autonomous loops perceive environment and execute tools.",
            metadata={"chunk_id": "page_01_chunk_01", "page": 1, "source": "Ebook-Agentic-AI.pdf"},
        ),
        DocumentChunk(
            chunk_id="page_02_chunk_01",
            source="Ebook-Agentic-AI.pdf",
            page=2,
            text="Agentic RAG evaluates relevance before generation.",
            metadata={"chunk_id": "page_02_chunk_01", "page": 2, "source": "Ebook-Agentic-AI.pdf"},
        ),
    ]


@pytest.fixture
def sample_embedded_chunks(sample_chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
    """Sample embedded chunks."""
    return [
        EmbeddedChunk(
            chunk_id=c.chunk_id,
            source=c.source,
            page=c.page,
            text=c.text,
            embedding=[0.1] * 512,
            metadata=c.metadata,
        )
        for c in sample_chunks
    ]


def test_successful_ingestion_flow(
    sample_pages: list[DocumentPage],
    sample_chunks: list[DocumentChunk],
    sample_embedded_chunks: list[EmbeddedChunk],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test complete repeatable ingestion flow and verify 6 progress stages."""
    mock_loader = MagicMock()
    mock_loader.load.return_value = sample_pages

    mock_chunker = MagicMock()
    mock_chunker.chunk_documents.return_value = sample_chunks

    mock_embedder = MagicMock()
    mock_embedder.embed_chunks.return_value = sample_embedded_chunks

    mock_store = MagicMock()
    mock_store.upsert_chunks.return_value = len(sample_embedded_chunks)

    summary = run_ingestion(
        pdf_path="data/Ebook-Agentic-AI.pdf",
        batch_size=50,
        namespace="custom_ns",
        loader=mock_loader,
        chunker=mock_chunker,
        embedder=mock_embedder,
        vector_store=mock_store,
    )

    captured = capsys.readouterr().out

    # 1. Verify progress stages printed
    assert "[1/6] Loading PDF" in captured
    assert "PDF loaded successfully" in captured
    assert "[2/6] Number of pages extracted: 2 non-empty page(s)" in captured
    assert "[3/6] Chunking document" in captured
    assert "Number of chunks generated: 2 chunk(s)" in captured
    assert "[4/6] Generating embeddings" in captured
    assert "Embeddings generated: 2 vector(s)" in captured
    assert "[5/6] Uploading vectors to Pinecone" in captured
    assert "Pinecone vectors uploaded: 2 vector(s)" in captured
    assert "[6/6] Ingestion completed successfully!" in captured

    # 2. Verify summary object
    assert isinstance(summary, IngestionSummary)
    assert summary.pages_count == 2
    assert summary.chunks_count == 2
    assert summary.vectors_uploaded == 2
    assert summary.namespace == "custom_ns"

    # 3. Verify component interactions
    mock_loader.load.assert_called_once()
    mock_chunker.chunk_documents.assert_called_once_with(sample_pages)
    mock_embedder.embed_chunks.assert_called_once_with(sample_chunks)
    mock_store.upsert_chunks.assert_called_once_with(
        chunks=sample_embedded_chunks,
        batch_size=50,
        namespace="custom_ns",
    )


def test_deterministic_chunk_ids_repeatable(
    sample_pages: list[DocumentPage],
    sample_chunks: list[DocumentChunk],
    sample_embedded_chunks: list[EmbeddedChunk],
) -> None:
    """Test that running ingestion repeatedly yields identical vector IDs."""
    mock_loader = MagicMock()
    mock_loader.load.return_value = sample_pages

    mock_chunker = MagicMock()
    mock_chunker.chunk_documents.return_value = sample_chunks

    mock_embedder = MagicMock()
    mock_embedder.embed_chunks.return_value = sample_embedded_chunks

    mock_store = MagicMock()
    mock_store.upsert_chunks.return_value = len(sample_embedded_chunks)

    # First run
    run_1 = run_ingestion(
        loader=mock_loader,
        chunker=mock_chunker,
        embedder=mock_embedder,
        vector_store=mock_store,
    )

    # Second run
    run_2 = run_ingestion(
        loader=mock_loader,
        chunker=mock_chunker,
        embedder=mock_embedder,
        vector_store=mock_store,
    )

    assert run_1.chunks_count == run_2.chunks_count
    assert run_1.vectors_uploaded == run_2.vectors_uploaded

    # First call upserted chunks IDs vs second call
    first_call_chunks = mock_store.upsert_chunks.call_args_list[0][1]["chunks"]
    second_call_chunks = mock_store.upsert_chunks.call_args_list[1][1]["chunks"]

    first_ids = [c.chunk_id for c in first_call_chunks]
    second_ids = [c.chunk_id for c in second_call_chunks]

    assert first_ids == second_ids
    assert first_ids == ["page_01_chunk_01", "page_02_chunk_01"]


def test_missing_pdf_raises_error() -> None:
    """Test that non-existent PDF file produces PDFNotFoundError and exits with 1 in main."""
    mock_loader = MagicMock()
    mock_loader.load.side_effect = PDFNotFoundError("PDF file 'non_existent.pdf' does not exist.")

    with pytest.raises(PDFNotFoundError, match="does not exist"):
        run_ingestion(pdf_path="non_existent.pdf", loader=mock_loader)

    # Test CLI exit code 1
    with patch("sys.argv", ["ingest.py", "--pdf", "non_existent.pdf"]), patch("scripts.ingest.run_ingestion", side_effect=PDFNotFoundError("Missing")):
        exit_code = main()
        assert exit_code == 1


def test_empty_pdf_raises_error() -> None:
    """Test that PDF with 0 extractable pages raises EmptyPDFError."""
    mock_loader = MagicMock()
    mock_loader.load.return_value = []

    with pytest.raises(EmptyPDFError, match="has no extractable text"):
        run_ingestion(loader=mock_loader)


def test_vector_store_failure_handled() -> None:
    """Test vector database error handling in CLI main()."""
    with patch("sys.argv", ["ingest.py"]), patch("scripts.ingest.run_ingestion", side_effect=VectorStoreError("Pinecone connection lost")):
        exit_code = main()
        assert exit_code == 1
