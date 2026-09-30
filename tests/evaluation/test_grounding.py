"""Evaluation test suite for answer groundedness and anti-hallucination verification.

Validates:
1. Grounded answer verification (facts supported by retrieved context).
2. Detection of unsupported / hallucinated claims (even if factually true in the real world).
3. LangGraph workflow loop behavior:
   - Single grounded attempt -> finalized immediately
   - Failed first attempt -> regenerate once -> success -> finalized
   - Failed first attempt + failed retry -> safe refusal
4. Empty context handling.
5. Response contract validation for all grounding outcomes:
   - query exists
   - final_answer exists
   - retrieved_context_chunks exists
   - confidence_score exists and is between 0 and 1
"""

from unittest.mock import MagicMock
import pytest

from app.generation.generator import FALLBACK_RESPONSE
from app.grading.groundedness import GroundednessEvaluator, GroundednessResult
from app.graph.workflow import run_rag
from app.retrieval.retriever import RetrievedChunk


@pytest.fixture
def sample_context_chunk() -> RetrievedChunk:
    """Fixture providing a sample retrieved chunk on Agentic AI memory."""
    return RetrievedChunk(
        chunk_id="page_10_chunk_01",
        source="Ebook-Agentic-AI.pdf",
        page=10,
        text=(
            "Agentic AI memory systems are partitioned into short-term working memory "
            "and long-term episodic or semantic memory stored in vector databases."
        ),
        similarity_score=0.92,
        metadata={"source": "Ebook-Agentic-AI.pdf", "page": 10, "chunk_id": "page_10_chunk_01"},
    )


def test_fully_grounded_answer_accepted_first_try(sample_context_chunk: RetrievedChunk) -> None:
    """Test that a fully grounded draft answer is accepted on the first attempt without retry."""
    query = "How is memory partitioned in Agentic AI?"
    grounded_answer = "Agentic AI memory is divided into short-term working memory and long-term vector memory."

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [sample_context_chunk]

    mock_generator = MagicMock()
    mock_generator.generate.return_value = grounded_answer

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate.return_value = GroundednessResult(
        grounded=True,
        grounding_score=0.96,
        unsupported_claims=[],
    )

    result = run_rag(
        query=query,
        retriever_service=mock_retriever,
        generator_service=mock_generator,
        groundedness_evaluator=mock_evaluator,
    )

    # Contract checks
    assert result.query == query
    assert result.final_answer == grounded_answer
    assert result.grounded is True
    assert result.grounding_score == 0.96
    assert result.confidence_score is not None
    assert 0.0 <= result.confidence_score <= 1.0
    assert result.confidence_score >= 0.70
    assert result.retry_count == 0
    assert len(result.retrieved_context) == 1

    # Verify evaluator called once
    mock_evaluator.evaluate.assert_called_once()


def test_real_world_fact_unsupported_by_document_is_marked_ungrounded() -> None:
    """Critical anti-hallucination rule: External real-world facts not in context must be marked ungrounded."""
    context = ["Agentic AI uses tool execution interfaces to connect to external systems."]
    draft_answer = (
        "Agentic AI uses tool interfaces. Furthermore, Python was created by Guido van Rossum in 1991."
    )

    evaluator = GroundednessEvaluator.__new__(GroundednessEvaluator)
    evaluator.client = MagicMock()
    evaluator.structured_client = MagicMock()

    # The LLM evaluator correctly identifies that Guido van Rossum is not in the context
    mock_eval_response = GroundednessResult(
        grounded=False,
        grounding_score=0.45,
        unsupported_claims=["Python was created by Guido van Rossum in 1991"],
    )
    evaluator.structured_client.invoke.return_value = mock_eval_response

    result = evaluator.evaluate(
        query="What tools does Agentic AI use?",
        draft_answer=draft_answer,
        retrieved_context=context,
    )

    assert result.grounded is False
    assert result.grounding_score < 0.50
    assert len(result.unsupported_claims) == 1
    assert "Guido van Rossum" in result.unsupported_claims[0]


def test_ungrounded_first_attempt_recovers_on_retry(sample_context_chunk: RetrievedChunk) -> None:
    """Test workflow self-correction: First ungrounded attempt triggers regenerate, second attempt succeeds."""
    query = "How is memory structured in Agentic AI?"
    hallucinated_draft = "Memory uses quantum entanglement modules to store thoughts forever."
    grounded_retry = "Memory is divided into short-term working memory and long-term vector storage."

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [sample_context_chunk]

    mock_generator = MagicMock()
    # First call yields hallucination, second call yields grounded text
    mock_generator.generate.side_effect = [hallucinated_draft, grounded_retry]

    mock_evaluator = MagicMock()
    # First eval fails, second eval passes
    mock_evaluator.evaluate.side_effect = [
        GroundednessResult(grounded=False, grounding_score=0.20, unsupported_claims=["quantum entanglement"]),
        GroundednessResult(grounded=True, grounding_score=0.94, unsupported_claims=[]),
    ]

    result = run_rag(
        query=query,
        retriever_service=mock_retriever,
        generator_service=mock_generator,
        groundedness_evaluator=mock_evaluator,
    )

    # Contract checks
    assert result.query == query
    assert result.final_answer == grounded_retry
    assert result.grounded is True
    assert result.grounding_score == 0.94
    assert result.confidence_score is not None
    assert 0.0 <= result.confidence_score <= 1.0
    assert result.confidence_score >= 0.70
    assert result.retry_count == 1

    # Verify both generate calls occurred
    assert mock_generator.generate.call_count == 2
    assert mock_evaluator.evaluate.call_count == 2


def test_repeatedly_ungrounded_answer_refused_after_max_retry(sample_context_chunk: RetrievedChunk) -> None:
    """Test that if regeneration still fails grounding check, workflow safely refuses."""
    query = "How is memory structured in Agentic AI?"
    hallucinated_draft_1 = "Memory relies on magical crystals."
    hallucinated_draft_2 = "Memory operates via telepathic waves."

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [sample_context_chunk]

    mock_generator = MagicMock()
    mock_generator.generate.side_effect = [hallucinated_draft_1, hallucinated_draft_2]

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate.side_effect = [
        GroundednessResult(grounded=False, grounding_score=0.10, unsupported_claims=["magical crystals"]),
        GroundednessResult(grounded=False, grounding_score=0.15, unsupported_claims=["telepathic waves"]),
    ]

    result = run_rag(
        query=query,
        retriever_service=mock_retriever,
        generator_service=mock_generator,
        groundedness_evaluator=mock_evaluator,
    )

    # Workflow must refuse with fallback response
    assert result.query == query
    assert result.final_answer == FALLBACK_RESPONSE
    assert result.grounded is False
    assert result.grounding_score == 0.0
    assert result.confidence_score == 0.0
    assert 0.0 <= result.confidence_score <= 1.0
    assert result.retry_count == 1

    # Generation retried exactly once (max 1 retry), then stopped
    assert mock_generator.generate.call_count == 2
    assert mock_evaluator.evaluate.call_count == 2


def test_empty_context_grounding_evaluator() -> None:
    """Test that evaluating grounding with empty context automatically returns grounded=False."""
    evaluator = GroundednessEvaluator.__new__(GroundednessEvaluator)
    evaluator.client = MagicMock()
    evaluator.structured_client = MagicMock()

    result = evaluator.evaluate(
        query="What is agent planning?",
        draft_answer="Planning is decomposing tasks into steps.",
        retrieved_context=[],
    )

    assert result.grounded is False
    assert result.grounding_score == 0.0
    assert len(result.unsupported_claims) == 1
    # LLM was not called because empty context is an instant fail
    evaluator.structured_client.invoke.assert_not_called()
