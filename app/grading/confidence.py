"""Transparent multi-signal confidence scoring system for Agentic RAG.

Confidence Formula Documentation:
--------------------------------
The confidence score represents the overall factual reliability of the generated response.
Rather than prompting an LLM to guess a confidence number, this module calculates
confidence deterministically from three explicit signals:

1. Retrieval Relevance (default weight = 0.40):
   Quality of the retrieved context matches from Pinecone (similarity score of top chunks).
   Measures whether the knowledge base contains relevant information for the query.

2. Grounding Score (default weight = 0.40):
   Factual entailment score from the groundedness verification evaluator.
   Measures whether claims in the generated response are directly supported by retrieved context.

3. Answerability / Retrieval Quality (default weight = 0.20):
   Assesses whether the query was answerable given the available context depth and whether
   the answer is non-refusal. Refused queries or ungrounded answers receive an answerability score of 0.0.

Formula:
    raw_confidence = (retrieval_score * w_retrieval) +
                     (grounding_score * w_grounding) +
                     (answerability_score * w_answerability)

    confidence_score = round(clamp(raw_confidence, 0.0, 1.0), 2)

Special Rules:
- Refused / out-of-scope queries (e.g. fallback refusal response): confidence_score is forced to 0.0.
- Completely unsupported answers (grounded=False, grounding_score=0.0):
  answerability is forced to 0.0 and overall confidence is heavily penalized.
- Weak retrieval matches below threshold suppress the overall confidence score.
"""

from collections.abc import Sequence
import logging
from typing import Any
from pydantic import BaseModel, Field

from app.generation.generator import FALLBACK_RESPONSE

logger = logging.getLogger(__name__)

DEFAULT_WEIGHT_RETRIEVAL: float = 0.40
DEFAULT_WEIGHT_GROUNDING: float = 0.40
DEFAULT_WEIGHT_ANSWERABILITY: float = 0.20


class ConfidenceWeights(BaseModel):
    """Configurable weights for the confidence scoring formula."""

    retrieval: float = Field(default=DEFAULT_WEIGHT_RETRIEVAL, ge=0.0, le=1.0)
    grounding: float = Field(default=DEFAULT_WEIGHT_GROUNDING, ge=0.0, le=1.0)
    answerability: float = Field(default=DEFAULT_WEIGHT_ANSWERABILITY, ge=0.0, le=1.0)

    def normalized(self) -> tuple[float, float, float]:
        """Return normalized weights summing to 1.0."""
        total = self.retrieval + self.grounding + self.answerability
        if total <= 0:
            return (
                DEFAULT_WEIGHT_RETRIEVAL,
                DEFAULT_WEIGHT_GROUNDING,
                DEFAULT_WEIGHT_ANSWERABILITY,
            )
        return (
            self.retrieval / total,
            self.grounding / total,
            self.answerability / total,
        )


class ConfidenceBreakdown(BaseModel):
    """Detailed breakdown of individual confidence signals."""

    retrieval_score: float = Field(..., ge=0.0, le=1.0)
    grounding_score: float = Field(..., ge=0.0, le=1.0)
    answerability_score: float = Field(..., ge=0.0, le=1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    weights: dict[str, float] = Field(default_factory=dict)
    is_refused: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert breakdown to clean dictionary."""
        return self.model_dump()


class ConfidenceScorer:
    """Calculates deterministic confidence scores from explicit RAG signals."""

    def __init__(self, weights: ConfidenceWeights | None = None) -> None:
        """Initialize ConfidenceScorer with configurable weights.

        Args:
            weights: Optional custom ConfidenceWeights configuration.
        """
        self.weights = weights or ConfidenceWeights()

    def calculate(
        self,
        retrieval_score: float,
        grounding_score: float,
        answerability_score: float | None = None,
        is_refused: bool = False,
        grounded: bool | None = None,
    ) -> ConfidenceBreakdown:
        """Calculate confidence score from explicit signals.

        Args:
            retrieval_score: Semantic similarity/relevance score (0.0 to 1.0).
            grounding_score: Factual grounding score (0.0 to 1.0).
            answerability_score: Optional answerability/context quality score (0.0 to 1.0).
            is_refused: True if query was refused or out of scope.
            grounded: True if answer was verified grounded, False if unsupported claims exist.

        Returns:
            ConfidenceBreakdown: Complete breakdown with individual scores and final confidence.
        """
        w_r, w_g, w_a = self.weights.normalized()
        weights_dict = {
            "retrieval": round(w_r, 4),
            "grounding": round(w_g, 4),
            "answerability": round(w_a, 4),
        }

        # 1. Refused / Out-of-scope queries produce low/zero confidence
        if is_refused:
            logger.debug("Query marked refused; returning 0.0 confidence.")
            return ConfidenceBreakdown(
                retrieval_score=0.0,
                grounding_score=0.0,
                answerability_score=0.0,
                confidence_score=0.0,
                weights=weights_dict,
                is_refused=True,
            )

        # 2. Bound component inputs
        r_score = max(0.0, min(1.0, float(retrieval_score)))
        g_score = max(0.0, min(1.0, float(grounding_score)))

        # 3. Determine answerability score
        if answerability_score is not None:
            a_score = max(0.0, min(1.0, float(answerability_score)))
        else:
            # Derived answerability: if answer is ungrounded, answerability is 0.0
            if grounded is False:
                a_score = 0.0
            elif grounded is True:
                # High-quality grounded answers inherit strong answerability
                a_score = min(1.0, round((r_score + g_score) / 2.0, 2))
            else:
                a_score = round(min(r_score, g_score), 2)

        # If answer is explicitly marked ungrounded, penalize answerability
        if grounded is False:
            a_score = 0.0

        # 4. Weighted formula calculation
        raw_confidence = (r_score * w_r) + (g_score * w_g) + (a_score * w_a)
        final_confidence = round(max(0.0, min(1.0, raw_confidence)), 2)

        logger.debug(
            "Confidence calculated: %.2f (r=%.2f, g=%.2f, a=%.2f)",
            final_confidence,
            r_score,
            g_score,
            a_score,
        )

        return ConfidenceBreakdown(
            retrieval_score=round(r_score, 4),
            grounding_score=round(g_score, 4),
            answerability_score=round(a_score, 4),
            confidence_score=final_confidence,
            weights=weights_dict,
            is_refused=False,
        )

    def calculate_from_state(self, state: dict[str, Any]) -> ConfidenceBreakdown:
        """Calculate confidence breakdown directly from a RAGState dictionary.

        Args:
            state: RAGState dictionary.

        Returns:
            ConfidenceBreakdown: Confidence calculation result.
        """
        final_answer = state.get("final_answer") or state.get("draft_answer") or ""
        is_refused = (
            final_answer == FALLBACK_RESPONSE
            or state.get("retrieval_relevant") is False
            or not state.get("retrieved_context")
        )

        if is_refused:
            return self.calculate(
                retrieval_score=0.0,
                grounding_score=0.0,
                answerability_score=0.0,
                is_refused=True,
            )

        # Compute retrieval score from retrieval_scores list
        scores = state.get("retrieval_scores", [])
        if scores:
            retrieval_score = float(sum(scores) / len(scores))
        else:
            retrieval_score = 0.0

        grounding_score = float(state.get("grounding_score") or 0.0)
        grounded = state.get("grounded")

        return self.calculate(
            retrieval_score=retrieval_score,
            grounding_score=grounding_score,
            is_refused=False,
            grounded=grounded,
        )


def calculate_confidence(
    retrieval_score: float,
    grounding_score: float,
    answerability_score: float | None = None,
    is_refused: bool = False,
    grounded: bool | None = None,
    weights: ConfidenceWeights | None = None,
) -> float:
    """Calculate transparent confidence score rounded to two decimal places.

    Formula:
        confidence_score =
            retrieval_score * 0.40 +
            grounding_score * 0.40 +
            answerability_score * 0.20

    Args:
        retrieval_score: Semantic relevance score (0.0 to 1.0).
        grounding_score: Factual grounding score (0.0 to 1.0).
        answerability_score: Optional retrieval quality / answerability score (0.0 to 1.0).
        is_refused: Whether query was refused.
        grounded: Whether draft answer was verified grounded.
        weights: Optional custom signal weights.

    Returns:
        float: Calculated confidence score between 0.0 and 1.0, rounded to 2 decimal places.
    """
    scorer = ConfidenceScorer(weights=weights)
    breakdown = scorer.calculate(
        retrieval_score=retrieval_score,
        grounding_score=grounding_score,
        answerability_score=answerability_score,
        is_refused=is_refused,
        grounded=grounded,
    )
    return breakdown.confidence_score
