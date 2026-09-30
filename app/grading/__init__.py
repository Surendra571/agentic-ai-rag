"""Document and hallucination grading module.

Responsible for evaluating retrieval relevance, grounding/hallucination checks,
and answer quality.
"""

from app.grading.groundedness import (
    GROUNDEDNESS_SYSTEM_PROMPT,
    GroundednessAPIError,
    GroundednessError,
    GroundednessEvaluator,
    GroundednessResult,
    MissingAPIKeyError,
    check_groundedness,
    format_evaluator_prompt,
)

__all__ = [
    "GROUNDEDNESS_SYSTEM_PROMPT",
    "GroundednessAPIError",
    "GroundednessError",
    "GroundednessEvaluator",
    "GroundednessResult",
    "MissingAPIKeyError",
    "check_groundedness",
    "format_evaluator_prompt",
]
