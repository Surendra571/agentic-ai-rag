"""Unit tests for Pinecone vector storage and retrieval with mocked client."""

from unittest.mock import MagicMock
import pytest

from app.config import get_settings
from app.ingestion.embedder import EmbeddedChunk
from app.retrieval.vector_store import (
    IndexNotFoundError,
    PineconeConnectionError,
    PineconeVectorStore,
    SearchResult,
    VectorStoreQueryError,
    VectorStoreUpsertError,
)


@pytest.fixture
def sample_embedded_chunks() -> list[EmbeddedChunk]:
    """Provide sample embedded chunks for upsert tests."""
    return [
        EmbeddedChunk(
            chunk_id=f"page_12_chunk_{i:02d}",
            source="Ebook-Agentic-AI.pdf",
            page=12,
            text=f"Text content for chunk {i}",
            embedding=[0.01 * i, 0.02 * i, 0.03 * i],
            metadata={
                "source": "Ebook-Agentic-AI.pdf",
                "page": 12,
                "chunk_id": f"page_12_chunk_{i:02d}",
            },
        )
        for i in range(1, 6)
    ]


@pytest.fixture
def mock_pinecone_index() -> MagicMock:
    """Mock Pinecone Index instance."""
    mock_idx = MagicMock()
    mock_idx.upsert.return_value = {"upserted_count": 5}
    return mock_idx


@pytest.fixture
def mock_pinecone_client(mock_pinecone_index: MagicMock) -> MagicMock:
    """Mock Pinecone client instance."""
    mock_client = MagicMock()
    mock_client.has_index.return_value = True
    mock_client.Index.return_value = mock_pinecone_index
    return mock_client


def test_connection_configuration_defaults(mock_pinecone_client: MagicMock) -> None:
    """Test that PineconeVectorStore reads settings by default."""
    settings = get_settings()
    store = PineconeVectorStore(client=mock_pinecone_client)

    assert store.api_key == settings.PINECONE_API_KEY
    assert store.index_name == settings.PINECONE_INDEX_NAME
    assert store.namespace == settings.PINECONE_NAMESPACE


def test_missing_api_key_raises_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that missing Pinecone API key raises PineconeConnectionError."""
    monkeypatch.setenv("PINECONE_API_KEY", "")
    get_settings.cache_clear()

    with pytest.raises(PineconeConnectionError) as exc_info:
        PineconeVectorStore(api_key="")

    assert "Pinecone API key is missing" in str(exc_info.value)
    get_settings.cache_clear()


def test_verify_index_creates_when_missing(mock_pinecone_client: MagicMock) -> None:
    """Test that ensure_index creates the index when missing and create_if_missing is True."""
    mock_pinecone_client.has_index.return_value = False

    store = PineconeVectorStore(
        client=mock_pinecone_client,
        index_name="new-test-index",
    )
    store.ensure_index(dimension=1536, metric="cosine", create_if_missing=True)

    mock_pinecone_client.create_index.assert_called_once()
    args, kwargs = mock_pinecone_client.create_index.call_args
    assert kwargs["name"] == "new-test-index"
    assert kwargs["dimension"] == 1536
    assert kwargs["metric"] == "cosine"


def test_verify_index_raises_when_missing_and_create_false(
    mock_pinecone_client: MagicMock,
) -> None:
    """Test that ensure_index raises IndexNotFoundError when missing and create_if_missing is False."""
    mock_pinecone_client.has_index.return_value = False

    store = PineconeVectorStore(
        client=mock_pinecone_client,
        index_name="nonexistent-index",
    )

    with pytest.raises(IndexNotFoundError) as exc_info:
        store.ensure_index(create_if_missing=False)

    assert "does not exist" in str(exc_info.value)
    mock_pinecone_client.create_index.assert_not_called()


def test_upsert_behavior_and_batches(
    sample_embedded_chunks: list[EmbeddedChunk],
    mock_pinecone_client: MagicMock,
    mock_pinecone_index: MagicMock,
) -> None:
    """Test that chunks are upserted in batches and return total count."""
    store = PineconeVectorStore(
        client=mock_pinecone_client,
        index=mock_pinecone_index,
    )

    # 5 chunks with batch_size=2 results in 3 batches: [2, 2, 1]
    upserted = store.upsert_chunks(sample_embedded_chunks, batch_size=2)

    assert upserted == 5
    assert mock_pinecone_index.upsert.call_count == 3


def test_metadata_storage_format(
    sample_embedded_chunks: list[EmbeddedChunk],
    mock_pinecone_client: MagicMock,
    mock_pinecone_index: MagicMock,
) -> None:
    """Test that required metadata fields are stored in the vector payload."""
    store = PineconeVectorStore(
        client=mock_pinecone_client,
        index=mock_pinecone_index,
    )

    store.upsert_chunks(sample_embedded_chunks[:1])

    mock_pinecone_index.upsert.assert_called_once()
    _, kwargs = mock_pinecone_index.upsert.call_args
    vectors = kwargs["vectors"]
    assert len(vectors) == 1

    first_vec = vectors[0]
    assert first_vec["id"] == "page_12_chunk_01"
    assert first_vec["values"] == [0.01, 0.02, 0.03]

    meta = first_vec["metadata"]
    assert meta["chunk_id"] == "page_12_chunk_01"
    assert meta["source"] == "Ebook-Agentic-AI.pdf"
    assert meta["page"] == 12
    assert meta["text"] == "Text content for chunk 1"


def test_similarity_search_results_structure(
    mock_pinecone_client: MagicMock,
    mock_pinecone_index: MagicMock,
) -> None:
    """Test that query results are mapped into structured SearchResult objects."""
    match_payload = MagicMock()
    match_payload.id = "page_12_chunk_03"
    match_payload.score = 0.945
    match_payload.metadata = {
        "text": "Agentic systems execute autonomous loops.",
        "source": "Ebook-Agentic-AI.pdf",
        "page": 12,
        "chunk_id": "page_12_chunk_03",
    }

    mock_response = MagicMock()
    mock_response.matches = [match_payload]
    mock_pinecone_index.query.return_value = mock_response

    store = PineconeVectorStore(
        client=mock_pinecone_client,
        index=mock_pinecone_index,
    )

    results = store.similarity_search_by_vector(query_vector=[0.1, 0.2, 0.3], top_k=5)

    assert len(results) == 1
    res = results[0]
    assert isinstance(res, SearchResult)
    assert res.text == "Agentic systems execute autonomous loops."
    assert res.page == 12
    assert res.chunk_id == "page_12_chunk_03"
    assert res.similarity_score == 0.945
    assert res.metadata["source"] == "Ebook-Agentic-AI.pdf"


def test_configurable_top_k(
    mock_pinecone_client: MagicMock,
    mock_pinecone_index: MagicMock,
) -> None:
    """Test that configurable top_k is passed to the Pinecone query call."""
    mock_response = MagicMock()
    mock_response.matches = []
    mock_pinecone_index.query.return_value = mock_response

    store = PineconeVectorStore(
        client=mock_pinecone_client,
        index=mock_pinecone_index,
    )

    # Test custom top_k
    store.similarity_search_by_vector(query_vector=[0.1, 0.2], top_k=8)
    _, kwargs = mock_pinecone_index.query.call_args
    assert kwargs["top_k"] == 8

    # Test default top_k
    store.similarity_search_by_vector(query_vector=[0.1, 0.2])
    _, kwargs = mock_pinecone_index.query.call_args
    assert kwargs["top_k"] == 5


def test_error_handling_upsert_failure(
    sample_embedded_chunks: list[EmbeddedChunk],
    mock_pinecone_client: MagicMock,
    mock_pinecone_index: MagicMock,
) -> None:
    """Test that Pinecone upsert failures are caught and wrapped in VectorStoreUpsertError."""
    mock_pinecone_index.upsert.side_effect = RuntimeError("Pinecone 503 Service Unavailable")

    store = PineconeVectorStore(
        client=mock_pinecone_client,
        index=mock_pinecone_index,
    )

    with pytest.raises(VectorStoreUpsertError) as exc_info:
        store.upsert_chunks(sample_embedded_chunks)

    assert "Failed to upsert vectors into Pinecone" in str(exc_info.value)
    assert "Pinecone 503" in str(exc_info.value)


def test_error_handling_query_failure(
    mock_pinecone_client: MagicMock,
    mock_pinecone_index: MagicMock,
) -> None:
    """Test that query errors are caught and wrapped in VectorStoreQueryError."""
    mock_pinecone_index.query.side_effect = RuntimeError("Pinecone query timeout")

    store = PineconeVectorStore(
        client=mock_pinecone_client,
        index=mock_pinecone_index,
    )

    with pytest.raises(VectorStoreQueryError) as exc_info:
        store.similarity_search_by_vector(query_vector=[0.1, 0.2])

    assert "Pinecone similarity query failed" in str(exc_info.value)


def test_similarity_search_convenience_with_mock_embedding_service(
    mock_pinecone_client: MagicMock,
    mock_pinecone_index: MagicMock,
) -> None:
    """Test end-to-end convenience search that embeds query and queries vector store."""
    mock_embedding_service = MagicMock()
    mock_embedding_service.embed_query.return_value = [0.12, 0.34, 0.56]

    mock_match = MagicMock()
    mock_match.id = "page_01_chunk_01"
    mock_match.score = 0.98
    mock_match.metadata = {
        "text": "Grounding in Agentic RAG.",
        "page": 1,
        "source": "Ebook-Agentic-AI.pdf",
        "chunk_id": "page_01_chunk_01",
    }
    mock_response = MagicMock()
    mock_response.matches = [mock_match]
    mock_pinecone_index.query.return_value = mock_response

    store = PineconeVectorStore(
        client=mock_pinecone_client,
        index=mock_pinecone_index,
    )

    results = store.similarity_search(
        query="What is grounding?",
        embedding_service=mock_embedding_service,
        top_k=3,
    )

    mock_embedding_service.embed_query.assert_called_once_with("What is grounding?")
    assert len(results) == 1
    assert results[0].chunk_id == "page_01_chunk_01"
    assert results[0].similarity_score == 0.98
