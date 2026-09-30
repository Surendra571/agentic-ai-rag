"""Unit tests for the transparent confidence scoring system."""

import pytest

from app.generation.generator import FALLBACK_RESPONSE
from app.grading.confidence import (
    ConfidenceBreakdown,
    ConfidenceScorer,
    ConfidenceWeights,
    calculate_confidence,
)


def test_high_quality_grounded_response() -> None:
    """1. High-quality grounded response: Strong retrieval, full grounding, high answerability."""
    # retrieval = 0.95, grounding = 1.0, answerability = 1.0
    # Expected: 0.95*0.4 + 1.0*0.4 + 1.0*0.2 = 0.38 + 0.40 + 0.20 = 0.98
    score = calculate_confidence(
        retrieval_score=0.95,
        grounding_score=1.0,
        answerability_score=1.0,
        grounded=True,
    )

    assert score == 0.98
    assert score >= 0.85


def test_medium_quality_response() -> None:
    """2. Medium-quality response: Moderate retrieval and partial grounding."""
    # retrieval = 0.70, grounding = 0.75, answerability = 0.60
    # Expected: 0.70*0.4 + 0.75*0.4 + 0.60*0.2 = 0.28 + 0.30 + 0.12 = 0.70
    score = calculate_confidence(
        retrieval_score=0.70,
        grounding_score=0.75,
        answerability_score=0.60,
        grounded=True,
    )

    assert score == 0.70
    assert 0.50 <= score <= 0.80


def test_weak_retrieval() -> None:
    """3. Weak retrieval: Low similarity matches heavily suppress confidence."""
    # retrieval = 0.25, grounding = 0.50, answerability = 0.20
    # Expected: 0.25*0.4 + 0.50*0.4 + 0.20*0.2 = 0.10 + 0.20 + 0.04 = 0.34
    score = calculate_confidence(
        retrieval_score=0.25,
        grounding_score=0.50,
        answerability_score=0.20,
    )

    assert score == 0.34
    assert score < 0.50


def test_refused_query() -> None:
    """4. Refused query: Refusal or out-of-scope question produces 0.0 confidence."""
    score = calculate_confidence(
        retrieval_score=0.80,
        grounding_score=0.90,
        is_refused=True,
    )

    assert score == 0.0

    # Also test via breakdown
    scorer = ConfidenceScorer()
    breakdown = scorer.calculate(
        retrieval_score=0.85,
        grounding_score=0.95,
        is_refused=True,
    )
    assert breakdown.confidence_score == 0.0
    assert breakdown.is_refused is True
    assert breakdown.retrieval_score == 0.0
    assert breakdown.grounding_score == 0.0


def test_unsupported_answer() -> None:
    """5. Unsupported answer: Answer not supported by context receives heavily penalized confidence."""
    # retrieval = 0.85, grounding = 0.0, grounded = False -> answerability = 0.0
    # Expected: 0.85*0.4 + 0.0*0.4 + 0.0*0.2 = 0.34
    score = calculate_confidence(
        retrieval_score=0.85,
        grounding_score=0.0,
        grounded=False,
    )

    assert score == 0.34
    assert score < 0.50


def test_custom_weights_and_normalization() -> None:
    """Test custom non-default weights with normalization."""
    # Custom weights: retrieval=0.5, grounding=0.3, answerability=0.2 (sums to 1.0)
    custom_weights = ConfidenceWeights(
        retrieval=0.50,
        grounding=0.30,
        answerability=0.20,
    )
    score = calculate_confidence(
        retrieval_score=0.80,
        grounding_score=0.80,
        answerability_score=0.80,
        weights=custom_weights,
    )

    assert score == 0.80

    # Weights summing to 2.0 get normalized
    unnormalized = ConfidenceWeights(
        retrieval=1.0,
        grounding=1.0,
        answerability=0.0,
    )
    score_norm = calculate_confidence(
        retrieval_score=0.60,
        grounding_score=0.80,
        weights=unnormalized,
    )
    # Normalized: retrieval=0.5, grounding=0.5 -> 0.60*0.5 + 0.80*0.5 = 0.70
    assert score_norm == 0.70


def test_calculate_from_state() -> None:
    """Test calculate_from_state extracts signals from RAGState properly."""
    scorer = ConfidenceScorer()

    # Grounded state
    valid_state = {
        "query": "What is Agentic RAG?",
        "final_answer": "Agentic RAG integrates autonomous loops with retrieval.",
        "retrieved_context": ["Agentic RAG integrates autonomous loops with retrieval."],
        "retrieval_scores": [0.90, 0.80],  # average = 0.85
        "retrieval_relevant": True,
        "grounded": True,
        "grounding_score": 0.95,
    }
    breakdown = scorer.calculate_from_state(valid_state)
    assert breakdown.confidence_score > 0.80
    assert breakdown.is_refused is False

    # Refused state
    refused_state = {
        "query": "Unknown question",
        "final_answer": FALLBACK_RESPONSE,
        "retrieved_context": [],
        "retrieval_relevant": False,
        "grounded": False,
    }
    refused_breakdown = scorer.calculate_from_state(refused_state)
    assert refused_breakdown.confidence_score == 0.0
    assert refused_breakdown.is_refused is True


def test_confidence_breakdown_dictionary() -> None:
    """Test ConfidenceBreakdown serialization to dictionary."""
    scorer = ConfidenceScorer()
    breakdown = scorer.calculate(
        retrieval_score=0.80,
        grounding_score=0.90,
        answerability_score=0.85,
    )

    data = breakdown.to_dict()
    assert isinstance(data, dict)
    assert "retrieval_score" in data
    assert "grounding_score" in data
    assert "answerability_score" in data
    assert "confidence_score" in data
    assert "weights" in data
    assert data["confidence_score"] == 0.85
