"""Unit tests for OpenAI embedding layer with mocked provider."""

import logging
from unittest.mock import MagicMock
import pytest

from app.config import get_settings
from app.ingestion.chunker import DocumentChunk
from app.ingestion.embedder import (
    EmbeddedChunk,
    EmbeddingAPIError,
    EmbeddingService,
    MissingAPIKeyError,
    embed_chunks,
)


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="page_12_chunk_01",
            source="Ebook-Agentic-AI.pdf",
            page=12,
            text="Autonomous loops perceive context, plan actions, and retrieve information.",
            metadata={"source": "Ebook-Agentic-AI.pdf", "page": 12, "chapter": 1},
        ),
        DocumentChunk(
            chunk_id="page_12_chunk_02",
            source="Ebook-Agentic-AI.pdf",
            page=12,
            text="Agentic RAG evaluates retrieval relevance and mitigates hallucinations.",
            metadata={"source": "Ebook-Agentic-AI.pdf", "page": 12, "chapter": 1},
        ),
    ]


@pytest.fixture
def mock_embeddings_client() -> MagicMock:
    mock_client = MagicMock()
    mock_client.embed_documents.return_value = [
        [0.012, -0.045, 0.089] * 512,
        [-0.034, 0.071, -0.015] * 512,
    ]
    mock_client.embed_query.return_value = [0.055, -0.021, 0.043] * 512
    return mock_client


def test_embedding_service_initializes_from_config(mock_embeddings_client: MagicMock) -> None:
    settings = get_settings()
    service = EmbeddingService(embeddings_client=mock_embeddings_client)
    assert service.model == settings.OPENAI_EMBEDDING_MODEL
    assert service.client == mock_embeddings_client


def test_missing_api_key_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    get_settings.cache_clear()

    with pytest.raises(MissingAPIKeyError) as exc_info:
        EmbeddingService(api_key="")

    assert "OpenAI API key is missing" in str(exc_info.value)
    get_settings.cache_clear()


def test_chunks_passed_to_embedding_layer(
    sample_chunks: list[DocumentChunk],
    mock_embeddings_client: MagicMock,
) -> None:
    service = EmbeddingService(embeddings_client=mock_embeddings_client)
    service.embed_chunks(sample_chunks)
    expected_texts = [chunk.text for chunk in sample_chunks]
    mock_embeddings_client.embed_documents.assert_called_once_with(expected_texts)


def test_returned_vectors_structure(
    sample_chunks: list[DocumentChunk],
    mock_embeddings_client: MagicMock,
) -> None:
    service = EmbeddingService(embeddings_client=mock_embeddings_client)
    embedded_chunks = service.embed_chunks(sample_chunks)
    assert len(embedded_chunks) == len(sample_chunks)
    for embedded in embedded_chunks:
        assert isinstance(embedded, EmbeddedChunk)
        assert isinstance(embedded.embedding, list)
        assert len(embedded.embedding) == 1536
        assert all(isinstance(v, float) for v in embedded.embedding)


def test_metadata_remains_associated(
    sample_chunks: list[DocumentChunk],
    mock_embeddings_client: MagicMock,
) -> None:
    service = EmbeddingService(embeddings_client=mock_embeddings_client)
    embedded_chunks = service.embed_chunks(sample_chunks)

    for original, embedded in zip(sample_chunks, embedded_chunks):
        assert embedded.chunk_id == original.chunk_id
        assert embedded.source == original.source
        assert embedded.page == original.page
        assert embedded.text == original.text
        assert embedded.metadata["chunk_id"] == original.chunk_id
        assert embedded.metadata["source"] == original.source
        assert embedded.metadata["page"] == original.page
        assert embedded.metadata.get("chapter") == 1


def test_api_errors_handled_cleanly(
    sample_chunks: list[DocumentChunk],
    mock_embeddings_client: MagicMock,
) -> None:
    mock_embeddings_client.embed_documents.side_effect = RuntimeError("OpenAI 429 Too Many Requests")
    service = EmbeddingService(embeddings_client=mock_embeddings_client)

    with pytest.raises(EmbeddingAPIError) as exc_info:
        service.embed_chunks(sample_chunks)

    assert "OpenAI embedding generation failed" in str(exc_info.value)
    assert "OpenAI 429" in str(exc_info.value)


def test_embed_query_success_and_error(mock_embeddings_client: MagicMock) -> None:
    service = EmbeddingService(embeddings_client=mock_embeddings_client)
    vector = service.embed_query("What is Agentic RAG?")
    assert len(vector) == 1536
    mock_embeddings_client.embed_query.assert_called_once_with("What is Agentic RAG?")

    with pytest.raises(ValueError):
        service.embed_query("   ")


def test_empty_chunks_returns_empty_list(mock_embeddings_client: MagicMock) -> None:
    service = EmbeddingService(embeddings_client=mock_embeddings_client)
    result = service.embed_chunks([])
    assert result == []
    mock_embeddings_client.embed_documents.assert_not_called()


def test_never_log_api_key(
    caplog: pytest.LogCaptureFixture,
    mock_embeddings_client: MagicMock,
) -> None:
    secret_key = "sk-super-confidential-secret-key-12345"
    with caplog.at_level(logging.DEBUG):
        EmbeddingService(
            api_key=secret_key,
            embeddings_client=mock_embeddings_client,
        )

    for record in caplog.records:
        assert secret_key not in record.message
