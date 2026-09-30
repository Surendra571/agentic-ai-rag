"""LangGraph state schema definition for Agentic RAG workflow."""

import json
from typing import Any, TypedDict


class RAGState(TypedDict, total=False):
    """Typed state for the Agentic RAG graph workflow.

    Maintains data across retrieval, grading, generation, and verification nodes.
    Contains only primitive, JSON-serializable data and no service or client instances.

    Fields:
        query: User input query or question.
        retrieved_context: Text content of retrieved candidate chunks.
        retrieval_scores: Relevance/similarity scores of retrieved context.
        retrieval_relevant: Evaluation flag indicating whether context is relevant.
        draft_answer: Candidate LLM answer generated from retrieved context.
        grounded: Evaluation flag indicating whether draft answer is factually grounded.
        grounding_score: Numeric score (0.0 - 1.0) indicating grounding alignment.
        confidence_score: Overall confidence score (0.0 - 1.0) for the response.
        final_answer: Validated and grounded final answer for the user.
        sources: Source citations metadata (document, page, chunk_id).
        retry_count: Number of generation retries performed.
    """

    query: str
    retrieved_context: list[str]
    retrieval_scores: list[float]
    retrieval_relevant: bool | None
    draft_answer: str | None
    grounded: bool | None
    grounding_score: float | None
    confidence_score: float | None
    final_answer: str | None
    sources: list[dict[str, Any]]
    retry_count: int


def create_initial_state(query: str) -> RAGState:
    """Initialize a clean, serializable RAGState for a new user query.

    Args:
        query: User's input question.

    Returns:
        RAGState: Initialized state dictionary.

    Raises:
        ValueError: If query is empty or whitespace-only.
    """
    if not isinstance(query, str):
        raise ValueError(f"Query must be a string, got {type(query).__name__}")

    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Query string cannot be empty or whitespace-only.")

    return {
        "query": clean_query,
        "retrieved_context": [],
        "retrieval_scores": [],
        "retrieval_relevant": None,
        "draft_answer": None,
        "grounded": None,
        "grounding_score": None,
        "confidence_score": None,
        "final_answer": None,
        "sources": [],
        "retry_count": 0,
    }


def is_state_serializable(state: RAGState) -> bool:
    """Verify that a state dictionary can be serialized to JSON without error."""
    try:
        json.dumps(state)
        return True
    except (TypeError, ValueError):
        return False
