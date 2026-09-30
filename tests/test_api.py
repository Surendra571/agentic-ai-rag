"""HTTP client integration tests for FastAPI chat endpoint."""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_rag_pipeline
from app.generation.generator import FALLBACK_RESPONSE, GenerationAPIError
from app.graph.workflow import RAGResult
from app.main import app
from app.retrieval.vector_store import VectorStoreError

client = TestClient(app)


@pytest.fixture
def mock_successful_rag_result() -> RAGResult:
    """Fixture providing a realistic grounded RAGResult."""
    return RAGResult(
        query="What is Agentic AI?",
        final_answer="Agentic AI systems operate through autonomous perception, reasoning, and tool execution loops.",
        grounded=True,
        grounding_score=0.96,
        confidence_score=0.94,
        retrieval_relevant=True,
        sources=[
            {"source": "Ebook-Agentic-AI.pdf", "page": 12, "chunk_id": "page_12_chunk_03"},
            {"source": "Ebook-Agentic-AI.pdf", "page": 14, "chunk_id": "page_14_chunk_01"},
        ],
        retrieved_context=[
            "Agentic AI systems operate through autonomous perception, reasoning, and tool execution loops.",
            "Agents evaluate and plan before acting.",
        ],
        retry_count=0,
    )


def test_post_chat_successful_response(mock_successful_rag_result: RAGResult) -> None:
    """Test successful POST /api/v1/chat with exact required JSON response structure."""
    mock_runner = MagicMock(return_value=mock_successful_rag_result)
    app.dependency_overrides[get_rag_pipeline] = lambda: mock_runner

    try:
        response = client.post(
            "/api/v1/chat",
            json={"query": "What is Agentic AI?"},
        )

        assert response.status_code == 200
        data = response.json()

        # 1. Verify all required fields
        assert "query" in data
        assert "final_answer" in data
        assert "retrieved_context_chunks" in data
        assert "confidence_score" in data

        # 2. Verify values match expectation
        assert data["query"] == "What is Agentic AI?"
        assert data["final_answer"] == mock_successful_rag_result.final_answer
        assert data["confidence_score"] == 0.94
        assert len(data["retrieved_context_chunks"]) == 2
        assert data["retrieved_context_chunks"] == mock_successful_rag_result.retrieved_context

        # 3. Verify additional allowed fields
        assert "grounded" in data
        assert data["grounded"] is True
        assert "grounding_score" in data
        assert data["grounding_score"] == 0.96
        assert "sources" in data
        assert len(data["sources"]) == 2

        # 4. Verify sources format
        first_source = data["sources"][0]
        assert first_source["page"] == 12
        assert first_source["chunk_id"] == "page_12_chunk_03"
        assert first_source["source"] == "Ebook-Agentic-AI.pdf"

        mock_runner.assert_called_once_with("What is Agentic AI?")
    finally:
        app.dependency_overrides.clear()


def test_post_chat_empty_query() -> None:
    """Test validation reject on empty or whitespace query."""
    # Empty string
    response_empty = client.post("/api/v1/chat", json={"query": ""})
    assert response_empty.status_code in (400, 422)

    # Whitespace string
    response_whitespace = client.post("/api/v1/chat", json={"query": "   \n\t  "})
    assert response_whitespace.status_code in (400, 422)


def test_post_chat_invalid_request_body() -> None:
    """Test validation reject when request body lacks required fields or is malformed."""
    # Missing query field
    response_missing = client.post("/api/v1/chat", json={})
    assert response_missing.status_code == 422

    # Wrong type
    response_wrong_type = client.post("/api/v1/chat", json={"query": 12345})
    assert response_wrong_type.status_code == 422


def test_post_chat_llm_error_handling() -> None:
    """Test that upstream LLM errors produce HTTP 502 with clean, sanitized message."""
    mock_runner = MagicMock(side_effect=GenerationAPIError("OpenAI API rate limit exceeded"))
    app.dependency_overrides[get_rag_pipeline] = lambda: mock_runner

    try:
        response = client.post(
            "/api/v1/chat",
            json={"query": "What is memory in agents?"},
        )

        assert response.status_code == 502
        data = response.json()
        assert "detail" in data
        assert "Language model" in data["detail"]
        # Ensure no internal error trace or API key leaks
        assert "rate limit exceeded" not in data["detail"]
        assert "OpenAI" not in data["detail"]
    finally:
        app.dependency_overrides.clear()


def test_post_chat_vector_db_error_handling() -> None:
    """Test that vector store errors produce HTTP 503 with clean, sanitized message."""
    mock_runner = MagicMock(side_effect=VectorStoreError("Pinecone connection lost"))
    app.dependency_overrides[get_rag_pipeline] = lambda: mock_runner

    try:
        response = client.post(
            "/api/v1/chat",
            json={"query": "Explain vector search"},
        )

        assert response.status_code == 503
        data = response.json()
        assert "detail" in data
        assert "Vector database" in data["detail"]
        assert "Pinecone" not in data["detail"]
    finally:
        app.dependency_overrides.clear()


def test_post_chat_unexpected_internal_error_handling() -> None:
    """Test unexpected exceptions produce HTTP 500 without leaking stack traces."""
    mock_runner = MagicMock(side_effect=RuntimeError("Secret database password failed in file /etc/secret"))
    app.dependency_overrides[get_rag_pipeline] = lambda: mock_runner

    try:
        response = client.post(
            "/api/v1/chat",
            json={"query": "Trigger internal error"},
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "internal error" in data["detail"].lower()
        assert "secret" not in data["detail"].lower()
        assert "/etc/secret" not in data["detail"]
    finally:
        app.dependency_overrides.clear()


def test_post_chat_refused_out_of_scope_query() -> None:
    """Test response structure when query is refused (no relevant context found)."""
    mock_refusal = RAGResult(
        query="What is quantum gravity in black holes?",
        final_answer=FALLBACK_RESPONSE,
        grounded=False,
        grounding_score=0.0,
        confidence_score=0.0,
        retrieval_relevant=False,
        sources=[],
        retrieved_context=[],
        retry_count=0,
    )
    mock_runner = MagicMock(return_value=mock_refusal)
    app.dependency_overrides[get_rag_pipeline] = lambda: mock_runner

    try:
        response = client.post(
            "/api/v1/chat",
            json={"query": "What is quantum gravity in black holes?"},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["final_answer"] == FALLBACK_RESPONSE
        assert data["confidence_score"] == 0.0
        assert data["grounded"] is False
        assert data["retrieved_context_chunks"] == []
        assert data["sources"] == []
    finally:
        app.dependency_overrides.clear()
