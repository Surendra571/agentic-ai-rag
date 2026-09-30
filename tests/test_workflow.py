"""Unit tests for the complete LangGraph Agentic RAG workflow."""

from unittest.mock import MagicMock
import pytest

from app.generation.generator import FALLBACK_RESPONSE, GenerationAPIError
from app.grading.groundedness import GroundednessResult
from app.graph.workflow import generate_mermaid_graph, run_rag
from app.retrieval.retriever import RetrievedChunk


@pytest.fixture
def mock_retriever() -> MagicMock:
    """Fixture providing a mock RetrieverService."""
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        RetrievedChunk(
            chunk_id="page_02_chunk_01",
            source="Ebook-Agentic-AI.pdf",
            page=2,
            text="Agentic AI systems operate through continuous perception, reasoning, and tool action loops.",
            similarity_score=0.91,
            metadata={"source": "Ebook-Agentic-AI.pdf", "page": 2, "chunk_id": "page_02_chunk_01"},
        )
    ]
    return retriever


@pytest.fixture
def mock_generator() -> MagicMock:
    """Fixture providing a mock GenerationService."""
    generator = MagicMock()
    generator.generate.return_value = (
        "Agentic AI systems operate through perception, reasoning, and tool action loops."
    )
    return generator


@pytest.fixture
def mock_evaluator() -> MagicMock:
    """Fixture providing a mock GroundednessEvaluator."""
    evaluator = MagicMock()
    evaluator.evaluate.return_value = GroundednessResult(
        grounded=True,
        grounding_score=0.95,
        unsupported_claims=[],
    )
    return evaluator


def test_successful_grounded_answer(
    mock_retriever: MagicMock,
    mock_generator: MagicMock,
    mock_evaluator: MagicMock,
) -> None:
    """1. Successful grounded answer: Retrieval, generation, and grounding all succeed."""
    result = run_rag(
        query="How do agentic systems operate?",
        retriever_service=mock_retriever,
        generator_service=mock_generator,
        groundedness_evaluator=mock_evaluator,
    )

    assert result.final_answer == (
        "Agentic AI systems operate through perception, reasoning, and tool action loops."
    )
    assert result.grounded is True
    assert result.grounding_score == 0.95
    assert result.confidence_score == 0.95
    assert result.retrieval_relevant is True
    assert result.retry_count == 0
    assert len(result.sources) == 1
    assert result.sources[0]["chunk_id"] == "page_02_chunk_01"


def test_no_retrieval(
    mock_retriever: MagicMock,
    mock_generator: MagicMock,
    mock_evaluator: MagicMock,
) -> None:
    """2. No retrieval: Missing documents trigger immediate refusal without calling LLM."""
    mock_retriever.retrieve.return_value = []

    result = run_rag(
        query="What is the capital of Mars?",
        retriever_service=mock_retriever,
        generator_service=mock_generator,
        groundedness_evaluator=mock_evaluator,
    )

    assert result.final_answer == FALLBACK_RESPONSE
    assert result.grounded is False
    assert result.retrieval_relevant is False
    assert result.sources == []
    # Generation and evaluator must NOT be called when retrieval is empty
    mock_generator.generate.assert_not_called()
    mock_evaluator.evaluate.assert_not_called()


def test_poor_retrieval(
    mock_retriever: MagicMock,
    mock_generator: MagicMock,
    mock_evaluator: MagicMock,
) -> None:
    """3. Poor retrieval: Low similarity matches filtered by retriever trigger refusal."""
    mock_retriever.retrieve.return_value = []

    result = run_rag(
        query="Irrelevant query below similarity threshold",
        retriever_service=mock_retriever,
        generator_service=mock_generator,
        groundedness_evaluator=mock_evaluator,
    )

    assert result.final_answer == FALLBACK_RESPONSE
    assert result.grounded is False
    assert result.retrieval_relevant is False
    mock_generator.generate.assert_not_called()
    mock_evaluator.evaluate.assert_not_called()


def test_grounded_first_attempt(
    mock_retriever: MagicMock,
    mock_generator: MagicMock,
    mock_evaluator: MagicMock,
) -> None:
    """4. Grounded first attempt: Workflow finalizes on first pass without triggering retry."""
    result = run_rag(
        query="Explain agent loops",
        retriever_service=mock_retriever,
        generator_service=mock_generator,
        groundedness_evaluator=mock_evaluator,
    )

    assert result.grounded is True
    assert result.retry_count == 0
    assert mock_generator.generate.call_count == 1
    assert mock_evaluator.evaluate.call_count == 1


def test_failed_first_grounding_successful_retry(
    mock_retriever: MagicMock,
    mock_generator: MagicMock,
    mock_evaluator: MagicMock,
) -> None:
    """5. Failed first grounding + successful retry: First draft is ungrounded; retry succeeds."""
    # First draft hallucinated; second draft grounded
    mock_generator.generate.side_effect = [
        "Hallucinated draft answer.",
        "Correctly grounded regenerated answer.",
    ]

    # First evaluation fails; second evaluation succeeds
    mock_evaluator.evaluate.side_effect = [
        GroundednessResult(grounded=False, grounding_score=0.30, unsupported_claims=["Hallucinated fact"]),
        GroundednessResult(grounded=True, grounding_score=0.92, unsupported_claims=[]),
    ]

    result = run_rag(
        query="Explain agent workflows",
        retriever_service=mock_retriever,
        generator_service=mock_generator,
        groundedness_evaluator=mock_evaluator,
    )

    assert result.final_answer == "Correctly grounded regenerated answer."
    assert result.grounded is True
    assert result.grounding_score == 0.92
    assert result.retry_count == 1
    assert mock_generator.generate.call_count == 2
    assert mock_evaluator.evaluate.call_count == 2


def test_failed_grounding_after_retry(
    mock_retriever: MagicMock,
    mock_generator: MagicMock,
    mock_evaluator: MagicMock,
) -> None:
    """6. Failed grounding after retry: Max 1 retry exceeded; routes safely to refusal."""
    mock_generator.generate.side_effect = [
        "First ungrounded draft.",
        "Second ungrounded draft.",
    ]

    mock_evaluator.evaluate.side_effect = [
        GroundednessResult(grounded=False, grounding_score=0.20, unsupported_claims=["Unverified claim 1"]),
        GroundednessResult(grounded=False, grounding_score=0.25, unsupported_claims=["Unverified claim 2"]),
    ]

    result = run_rag(
        query="Question leading to repeated hallucinations",
        retriever_service=mock_retriever,
        generator_service=mock_generator,
        groundedness_evaluator=mock_evaluator,
    )

    assert result.final_answer == FALLBACK_RESPONSE
    assert result.grounded is False
    assert result.retry_count == 1
    assert mock_generator.generate.call_count == 2
    assert mock_evaluator.evaluate.call_count == 2


def test_llm_failure(
    mock_retriever: MagicMock,
    mock_generator: MagicMock,
    mock_evaluator: MagicMock,
) -> None:
    """7. LLM failure: API errors in generation do not crash the workflow and result in safe refusal."""
    mock_generator.generate.side_effect = GenerationAPIError("LLM API service unavailable")

    result = run_rag(
        query="Valid question with broken LLM",
        retriever_service=mock_retriever,
        generator_service=mock_generator,
        groundedness_evaluator=mock_evaluator,
    )

    assert result.final_answer == FALLBACK_RESPONSE
    assert result.grounded is False


def test_generate_mermaid_graph() -> None:
    """Test Mermaid diagram generation for graph visualization."""
    diagram = generate_mermaid_graph()
    assert isinstance(diagram, str)
    assert len(diagram) > 0
    assert "retrieve" in diagram
    assert "grade_retrieval" in diagram
    assert "generate" in diagram
    assert "check_grounding" in diagram
    assert "refuse" in diagram
