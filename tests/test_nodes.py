"""Unit tests for LangGraph retrieve_node and generate_node with mocked services."""

from unittest.mock import MagicMock
import pytest

from app.generation.generator import FALLBACK_RESPONSE, GenerationAPIError
from app.graph.nodes import generate_node, retrieve_node
from app.graph.state import create_initial_state, is_state_serializable
from app.retrieval.retriever import RetrievedChunk


@pytest.fixture
def mock_retriever_service() -> MagicMock:
    """Mock RetrieverService."""
    return MagicMock()


@pytest.fixture
def mock_generator_service() -> MagicMock:
    """Mock GenerationService."""
    return MagicMock()


@pytest.fixture
def sample_retrieved_chunks() -> list[RetrievedChunk]:
    """Provide sample retrieved chunks for node testing."""
    return [
        RetrievedChunk(
            chunk_id="page_12_chunk_01",
            source="Ebook-Agentic-AI.pdf",
            page=12,
            text="Autonomous loops perceive context, plan actions, and retrieve information.",
            similarity_score=0.92,
            metadata={"source": "Ebook-Agentic-AI.pdf", "page": 12, "chunk_id": "page_12_chunk_01"},
        ),
        RetrievedChunk(
            chunk_id="page_14_chunk_02",
            source="Ebook-Agentic-AI.pdf",
            page=14,
            text="Agentic RAG verifies retrieval relevance before generating an answer.",
            similarity_score=0.85,
            metadata={"source": "Ebook-Agentic-AI.pdf", "page": 14, "chunk_id": "page_14_chunk_02"},
        ),
    ]


# ---------------------------------------------------------------------------
# retrieve_node Tests
# ---------------------------------------------------------------------------


def test_retrieve_node_successful(
    mock_retriever_service: MagicMock,
    sample_retrieved_chunks: list[RetrievedChunk],
) -> None:
    """Test successful retrieval node execution updates RAGState."""
    mock_retriever_service.retrieve.return_value = sample_retrieved_chunks

    state = create_initial_state("Explain agentic RAG workflows")
    update = retrieve_node(state, retriever_service=mock_retriever_service)

    mock_retriever_service.retrieve.assert_called_once_with("Explain agentic RAG workflows")

    assert update["retrieval_relevant"] is True
    assert len(update["retrieved_context"]) == 2
    assert update["retrieved_context"][0] == sample_retrieved_chunks[0].text
    assert update["retrieved_context"][1] == sample_retrieved_chunks[1].text
    assert update["retrieval_scores"] == [0.92, 0.85]
    assert len(update["sources"]) == 2

    # Check updated state serializability
    state.update(update)
    assert is_state_serializable(state) is True


def test_retrieve_node_no_results(mock_retriever_service: MagicMock) -> None:
    """Test retrieval node behavior when no matches exist."""
    mock_retriever_service.retrieve.return_value = []

    state = create_initial_state("Completely unrelated query")
    update = retrieve_node(state, retriever_service=mock_retriever_service)

    assert update["retrieval_relevant"] is False
    assert update["retrieved_context"] == []
    assert update["retrieval_scores"] == []
    assert update["sources"] == []


def test_retrieve_node_weak_retrieval(mock_retriever_service: MagicMock) -> None:
    """Test weak retrieval where all matches fail similarity thresholds."""
    mock_retriever_service.retrieve.return_value = []

    state = create_initial_state("Low quality query")
    update = retrieve_node(state, retriever_service=mock_retriever_service)

    assert update["retrieval_relevant"] is False


def test_retrieve_node_metadata_preservation(
    mock_retriever_service: MagicMock,
    sample_retrieved_chunks: list[RetrievedChunk],
) -> None:
    """Test that source document, page, and chunk_id metadata are preserved in state."""
    mock_retriever_service.retrieve.return_value = sample_retrieved_chunks

    state = create_initial_state("What is the architecture?")
    update = retrieve_node(state, retriever_service=mock_retriever_service)

    sources = update["sources"]
    assert len(sources) == 2

    first_source = sources[0]
    assert first_source["source"] == "Ebook-Agentic-AI.pdf"
    assert first_source["page"] == 12
    assert first_source["chunk_id"] == "page_12_chunk_01"

    second_source = sources[1]
    assert second_source["source"] == "Ebook-Agentic-AI.pdf"
    assert second_source["page"] == 14
    assert second_source["chunk_id"] == "page_14_chunk_02"


def test_retrieve_node_empty_query(mock_retriever_service: MagicMock) -> None:
    """Test that empty or whitespace query safely returns non-relevant without crashing."""
    state = {"query": "   "}
    update = retrieve_node(state, retriever_service=mock_retriever_service)  # type: ignore[arg-type]

    assert update["retrieval_relevant"] is False
    assert update["retrieved_context"] == []
    mock_retriever_service.retrieve.assert_not_called()


def test_retrieve_node_handles_service_exception(mock_retriever_service: MagicMock) -> None:
    """Test that unexpected retrieval service errors are caught cleanly."""
    mock_retriever_service.retrieve.side_effect = RuntimeError("Pinecone connection lost")

    state = create_initial_state("Valid query")
    update = retrieve_node(state, retriever_service=mock_retriever_service)

    assert update["retrieval_relevant"] is False
    assert update["retrieved_context"] == []
    assert update["retrieval_scores"] == []
    assert update["sources"] == []


# ---------------------------------------------------------------------------
# generate_node Tests
# ---------------------------------------------------------------------------


def test_generate_node_successful(mock_generator_service: MagicMock) -> None:
    """Test successful generation node execution populates draft_answer."""
    expected_answer = "Perception and action loops form the backbone of autonomous agents."
    mock_generator_service.generate.return_value = expected_answer

    state = create_initial_state("How do agents operate?")
    state["retrieved_context"] = ["Agents rely on perception and action loops."]
    state["retrieval_relevant"] = True
    state["sources"] = [{"source": "Ebook-Agentic-AI.pdf", "page": 4, "chunk_id": "page_04_chunk_01"}]

    update = generate_node(state, generator_service=mock_generator_service)

    assert update["draft_answer"] == expected_answer
    mock_generator_service.generate.assert_called_once_with(
        question="How do agents operate?",
        retrieved_context=["Agents rely on perception and action loops."],
        sources=[{"source": "Ebook-Agentic-AI.pdf", "page": 4, "chunk_id": "page_04_chunk_01"}],
    )

    state.update(update)
    assert is_state_serializable(state) is True


def test_generate_node_empty_query(mock_generator_service: MagicMock) -> None:
    """Test generate_node with empty query returns fallback without calling generator service."""
    state = {"query": "   ", "retrieved_context": ["Some context"]}
    update = generate_node(state, generator_service=mock_generator_service)  # type: ignore[arg-type]

    assert update["draft_answer"] == FALLBACK_RESPONSE
    mock_generator_service.generate.assert_not_called()


def test_generate_node_retrieval_not_relevant(mock_generator_service: MagicMock) -> None:
    """Test generate_node returns fallback if retrieval was marked irrelevant."""
    state = create_initial_state("Out of domain question")
    state["retrieved_context"] = ["Irrelevant context"]
    state["retrieval_relevant"] = False

    update = generate_node(state, generator_service=mock_generator_service)

    assert update["draft_answer"] == FALLBACK_RESPONSE
    mock_generator_service.generate.assert_not_called()


def test_generate_node_empty_context(mock_generator_service: MagicMock) -> None:
    """Test generate_node returns fallback if retrieved_context is empty."""
    state = create_initial_state("Question with no matches")
    state["retrieved_context"] = []
    state["retrieval_relevant"] = True

    update = generate_node(state, generator_service=mock_generator_service)

    assert update["draft_answer"] == FALLBACK_RESPONSE
    mock_generator_service.generate.assert_not_called()


def test_generate_node_handles_service_exception(mock_generator_service: MagicMock) -> None:
    """Test generate_node gracefully returns fallback if generation service throws exception."""
    mock_generator_service.generate.side_effect = GenerationAPIError("LLM API request timed out")

    state = create_initial_state("Valid query")
    state["retrieved_context"] = ["Valid document context"]
    state["retrieval_relevant"] = True

    update = generate_node(state, generator_service=mock_generator_service)

    assert update["draft_answer"] == FALLBACK_RESPONSE
