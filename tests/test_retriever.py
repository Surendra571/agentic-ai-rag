"""Unit tests for query retrieval service using mocked embeddings and vector store."""

from unittest.mock import MagicMock
import pytest

from app.config import get_settings
from app.retrieval.retriever import (
    RetrievalResponse,
    RetrievedChunk,
    RetrieverService,
)
from app.retrieval.vector_store import SearchResult


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    """Mock EmbeddingService with query embedding generation."""
    service = MagicMock()
    service.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
    return service


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Mock PineconeVectorStore."""
    store = MagicMock()
    return store


def test_relevant_query_returns_structured_results(
    mock_embedding_service: MagicMock,
    mock_vector_store: MagicMock,
) -> None:
    """Test retrieving relevant chunks with scores exceeding threshold."""
    mock_vector_store.similarity_search_by_vector.return_value = [
        SearchResult(
            text="Autonomous loops perceive context and plan actions.",
            page=12,
            chunk_id="page_12_chunk_01",
            similarity_score=0.88,
            metadata={"source": "Ebook-Agentic-AI.pdf", "page": 12, "chunk_id": "page_12_chunk_01"},
        ),
        SearchResult(
            text="Agentic RAG evaluates retrieval relevance and grounding.",
            page=12,
            chunk_id="page_12_chunk_02",
            similarity_score=0.76,
            metadata={"source": "Ebook-Agentic-AI.pdf", "page": 12, "chunk_id": "page_12_chunk_02"},
        ),
    ]

    retriever = RetrieverService(
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
        top_k=5,
        similarity_threshold=0.5,
    )

    chunks = retriever.retrieve("What is an autonomous agent?")

    assert len(chunks) == 2
    mock_embedding_service.embed_query.assert_called_once_with("What is an autonomous agent?")
    mock_vector_store.similarity_search_by_vector.assert_called_once()

    first = chunks[0]
    assert isinstance(first, RetrievedChunk)
    assert first.chunk_id == "page_12_chunk_01"
    assert first.page == 12
    assert first.source == "Ebook-Agentic-AI.pdf"
    assert first.similarity_score == 0.88
    assert "Autonomous loops" in first.text


def test_no_results_returns_empty_list(
    mock_embedding_service: MagicMock,
    mock_vector_store: MagicMock,
) -> None:
    """Test that empty vector store results return an empty list without inventing context."""
    mock_vector_store.similarity_search_by_vector.return_value = []

    retriever = RetrieverService(
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
    )

    chunks = retriever.retrieve("Unrelated esoteric query")
    assert chunks == []


def test_low_similarity_matches_filtered(
    mock_embedding_service: MagicMock,
    mock_vector_store: MagicMock,
) -> None:
    """Test that results below the similarity threshold are filtered out."""
    mock_vector_store.similarity_search_by_vector.return_value = [
        SearchResult(
            text="Highly relevant chunk about agent loops.",
            page=14,
            chunk_id="page_14_chunk_01",
            similarity_score=0.82,
            metadata={"source": "Ebook-Agentic-AI.pdf", "page": 14},
        ),
        SearchResult(
            text="Borderline low quality match.",
            page=20,
            chunk_id="page_20_chunk_01",
            similarity_score=0.48,  # Below threshold 0.50
            metadata={"source": "Ebook-Agentic-AI.pdf", "page": 20},
        ),
        SearchResult(
            text="Irrelevant noise match.",
            page=45,
            chunk_id="page_45_chunk_01",
            similarity_score=0.21,  # Far below threshold
            metadata={"source": "Ebook-Agentic-AI.pdf", "page": 45},
        ),
    ]

    retriever = RetrieverService(
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
        similarity_threshold=0.50,
    )

    chunks = retriever.retrieve("How do agent loops work?")

    # Only the first chunk should qualify
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "page_14_chunk_01"
    assert chunks[0].similarity_score == 0.82


def test_all_chunks_below_threshold_returns_empty(
    mock_embedding_service: MagicMock,
    mock_vector_store: MagicMock,
) -> None:
    """Test that when all candidate matches are low similarity, an empty result is returned."""
    mock_vector_store.similarity_search_by_vector.return_value = [
        SearchResult(
            text="Weak similarity.",
            page=1,
            chunk_id="page_01_chunk_01",
            similarity_score=0.35,
            metadata={"source": "Ebook-Agentic-AI.pdf", "page": 1},
        ),
    ]

    retriever = RetrieverService(
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
        similarity_threshold=0.60,
    )

    chunks = retriever.retrieve("Any query")
    assert chunks == []


def test_configurable_top_k(
    mock_embedding_service: MagicMock,
    mock_vector_store: MagicMock,
) -> None:
    """Test that top_k can be configured at initialization and per query."""
    mock_vector_store.similarity_search_by_vector.return_value = []

    retriever = RetrieverService(
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
        top_k=5,
    )

    # Custom per-query top_k
    retriever.retrieve("What is RAG?", top_k=3)
    _, kwargs = mock_vector_store.similarity_search_by_vector.call_args
    assert kwargs["top_k"] == 3

    # Default top_k
    retriever.retrieve("What is RAG?")
    _, kwargs = mock_vector_store.similarity_search_by_vector.call_args
    assert kwargs["top_k"] == 5


def test_metadata_preservation(
    mock_embedding_service: MagicMock,
    mock_vector_store: MagicMock,
) -> None:
    """Test that page, chunk_id, source, text, and custom metadata are intact."""
    meta = {
        "source": "Ebook-Agentic-AI.pdf",
        "page": 15,
        "chunk_id": "page_15_chunk_03",
        "section": "Hallucination Mitigation",
    }
    mock_vector_store.similarity_search_by_vector.return_value = [
        SearchResult(
            text="Hallucination grading evaluates factual grounding.",
            page=15,
            chunk_id="page_15_chunk_03",
            similarity_score=0.91,
            metadata=meta,
        )
    ]

    retriever = RetrieverService(
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
    )

    chunks = retriever.retrieve("How is hallucination graded?")
    assert len(chunks) == 1
    chunk = chunks[0]

    assert chunk.chunk_id == "page_15_chunk_03"
    assert chunk.page == 15
    assert chunk.source == "Ebook-Agentic-AI.pdf"
    assert chunk.text == "Hallucination grading evaluates factual grounding."
    assert chunk.similarity_score == 0.91
    assert chunk.metadata["section"] == "Hallucination Mitigation"


@pytest.mark.parametrize(
    "bad_query",
    [
        "",
        "   ",
        "\t\n  \n",
        12345,
        None,
    ],
)
def test_invalid_query_raises_value_error(
    bad_query: object,
    mock_embedding_service: MagicMock,
    mock_vector_store: MagicMock,
) -> None:
    """Test that empty or non-string queries raise ValueError."""
    retriever = RetrieverService(
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
    )

    with pytest.raises(ValueError):
        retriever.retrieve(bad_query)  # type: ignore[arg-type]


def test_retrieve_with_details(
    mock_embedding_service: MagicMock,
    mock_vector_store: MagicMock,
) -> None:
    """Test detailed retrieval response wrapper."""
    mock_vector_store.similarity_search_by_vector.return_value = [
        SearchResult(
            text="Detailed retrieval output test.",
            page=1,
            chunk_id="page_01_chunk_01",
            similarity_score=0.85,
            metadata={"source": "Ebook-Agentic-AI.pdf", "page": 1},
        )
    ]

    retriever = RetrieverService(
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
        top_k=4,
        similarity_threshold=0.6,
    )

    response = retriever.retrieve_with_details("Explain agentic architecture")

    assert isinstance(response, RetrievalResponse)
    assert response.query == "Explain agentic architecture"
    assert response.top_k == 4
    assert response.similarity_threshold == 0.6
    assert len(response.chunks) == 1
    assert response.chunks[0].chunk_id == "page_01_chunk_01"
