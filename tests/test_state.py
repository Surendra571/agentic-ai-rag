"""Unit tests for LangGraph RAGState definition and updates."""

import json
import pytest

from app.graph.state import (
    RAGState,
    create_initial_state,
    is_state_serializable,
)


def test_create_initial_state() -> None:
    """Test creating an initial state with a valid query."""
    query = "What are the core components of Agentic RAG?"
    state = create_initial_state(query)

    assert state["query"] == query
    assert state["retrieved_context"] == []
    assert state["retrieval_scores"] == []
    assert state["retrieval_relevant"] is None
    assert state["draft_answer"] is None
    assert state["grounded"] is None
    assert state["grounding_score"] is None
    assert state["confidence_score"] is None
    assert state["final_answer"] is None
    assert state["sources"] == []
    assert state["retry_count"] == 0


@pytest.mark.parametrize("invalid_query", ["", "   ", "\t\n  ", None, 123])
def test_create_initial_state_invalid_query_raises(invalid_query: object) -> None:
    """Test that invalid queries raise ValueError on state creation."""
    with pytest.raises(ValueError):
        create_initial_state(invalid_query)  # type: ignore[arg-type]


def test_state_lifecycle_updates() -> None:
    """Test progressive state updates through mock workflow stages."""
    state = create_initial_state("Explain agent loops.")

    # 1. Retrieval stage
    retrieved_texts = [
        "Agentic systems execute autonomous loops that perceive and plan.",
        "Retrieval is dynamic and verified before answer generation.",
    ]
    retrieved_scores = [0.89, 0.74]
    sources = [
        {"source": "Ebook-Agentic-AI.pdf", "page": 12, "chunk_id": "page_12_chunk_01"},
        {"source": "Ebook-Agentic-AI.pdf", "page": 12, "chunk_id": "page_12_chunk_02"},
    ]

    state["retrieved_context"] = retrieved_texts
    state["retrieval_scores"] = retrieved_scores
    state["sources"] = sources

    assert len(state["retrieved_context"]) == 2
    assert state["retrieval_scores"][0] == 0.89
    assert len(state["sources"]) == 2

    # 2. Retrieval grading stage
    state["retrieval_relevant"] = True
    assert state["retrieval_relevant"] is True

    # 3. Draft generation stage
    state["draft_answer"] = "Agent loops perceive context and take actions iteratively."
    assert state["draft_answer"] is not None

    # 4. Grounding evaluation stage
    state["grounded"] = True
    state["grounding_score"] = 0.95
    state["confidence_score"] = 0.92

    assert state["grounded"] is True
    assert state["grounding_score"] == 0.95
    assert state["confidence_score"] == 0.92

    # 5. Final answer stage
    state["final_answer"] = state["draft_answer"]
    assert state["final_answer"] == state["draft_answer"]


def test_state_is_strictly_json_serializable() -> None:
    """Test that fully populated state is cleanly serializable to JSON."""
    state: RAGState = {
        "query": "How is hallucination graded?",
        "retrieved_context": [
            "Hallucination grading evaluates factual overlap with retrieved context."
        ],
        "retrieval_scores": [0.93],
        "retrieval_relevant": True,
        "draft_answer": "Hallucination grading checks grounding against documents.",
        "grounded": True,
        "grounding_score": 0.98,
        "confidence_score": 0.95,
        "final_answer": "Hallucination grading checks grounding against documents.",
        "sources": [
            {
                "source": "Ebook-Agentic-AI.pdf",
                "page": 15,
                "chunk_id": "page_15_chunk_03",
            }
        ],
    }

    assert is_state_serializable(state) is True

    # Confirm json.dumps succeeds and json.loads reconstructs matching data
    serialized = json.dumps(state)
    deserialized = json.loads(serialized)
    assert deserialized == state


def test_unserializable_objects_detected() -> None:
    """Test that unserializable service instances or callables fail serialization check."""
    class DummyClient:
        pass

    bad_state: dict[str, object] = {
        "query": "test query",
        "client": DummyClient(),  # Not serializable
    }

    assert is_state_serializable(bad_state) is False  # type: ignore[arg-type]
