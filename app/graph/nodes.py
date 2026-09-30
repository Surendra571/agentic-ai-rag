"""LangGraph node definitions for the Agentic RAG workflow."""

import logging
from typing import Any

from app.generation.generator import FALLBACK_RESPONSE, GenerationService
from app.grading.groundedness import GroundednessEvaluator
from app.graph.state import RAGState
from app.retrieval.retriever import RetrieverService

logger = logging.getLogger(__name__)


def retrieve_node(
    state: RAGState,
    retriever_service: RetrieverService | None = None,
) -> dict[str, Any]:
    """LangGraph node to fetch grounded document chunks for the input query.

    Steps:
    1. Read state["query"].
    2. Call the existing RetrieverService.
    3. Store retrieved chunk texts in retrieved_context.
    4. Store similarity scores in retrieval_scores.
    5. Determine whether useful context exists and set retrieval_relevant.
    6. Store citation metadata in sources.

    Args:
        state: Current RAGState dictionary.
        retriever_service: Optional RetrieverService instance (defaults to production instance).

    Returns:
        dict[str, Any]: State update dictionary containing retrieval fields.
    """
    service = retriever_service or RetrieverService()

    raw_query = state.get("query", "")
    query = raw_query.strip() if isinstance(raw_query, str) else ""

    if not query:
        logger.warning("Empty query supplied to retrieve_node.")
        return {
            "retrieved_context": [],
            "retrieval_scores": [],
            "retrieval_relevant": False,
            "sources": [],
        }

    try:
        chunks = service.retrieve(query)
    except Exception as exc:
        logger.error("Failed executing retrieval for query '%s': %s", query, exc)
        return {
            "retrieved_context": [],
            "retrieval_scores": [],
            "retrieval_relevant": False,
            "sources": [],
        }

    has_relevant_context = len(chunks) > 0

    retrieved_context = [chunk.text for chunk in chunks]
    retrieval_scores = [chunk.similarity_score for chunk in chunks]
    sources = [
        {
            "source": chunk.source,
            "page": chunk.page,
            "chunk_id": chunk.chunk_id,
        }
        for chunk in chunks
    ]

    logger.info(
        "retrieve_node finished: found %d relevant chunks (relevant=%s)",
        len(chunks),
        has_relevant_context,
    )

    return {
        "retrieved_context": retrieved_context,
        "retrieval_scores": retrieval_scores,
        "retrieval_relevant": has_relevant_context,
        "sources": sources,
    }


def grade_retrieval_node(state: RAGState) -> dict[str, Any]:
    """Grade whether retrieved context is sufficient to proceed to generation.

    If retrieval_relevant is False or retrieved_context has no content,
    marks retrieval_relevant as False.

    Args:
        state: Current RAGState dictionary.

    Returns:
        dict[str, Any]: State update dictionary with 'retrieval_relevant'.
    """
    retrieved = state.get("retrieved_context", [])
    relevant_flag = state.get("retrieval_relevant")

    has_content = any(bool(t.strip()) for t in retrieved if isinstance(t, str))
    is_sufficient = bool(relevant_flag is True and has_content)

    logger.info(
        "grade_retrieval_node: is_sufficient=%s (chunks=%d)",
        is_sufficient,
        len(retrieved),
    )

    return {"retrieval_relevant": is_sufficient}


def decide_retrieval(state: RAGState) -> str:
    """Routing condition after grade_retrieval node.

    Returns:
        str: 'generate' if context is sufficient, else 'refuse'.
    """
    if state.get("retrieval_relevant") is True:
        return "generate"
    return "refuse"


def generate_node(
    state: RAGState,
    generator_service: GenerationService | None = None,
) -> dict[str, Any]:
    """LangGraph node to generate a draft answer grounded in retrieved context.

    Steps:
    1. Read state["query"].
    2. Read state["retrieved_context"] and state.get("sources", []).
    3. Check if query is empty, retrieval_relevant is False, or retrieved_context is empty.
       If so, return {"draft_answer": FALLBACK_RESPONSE}.
    4. Call generator_service.generate(question, retrieved_context, sources).
    5. Return {"draft_answer": draft_answer}.

    Args:
        state: Current RAGState dictionary.
        generator_service: Optional GenerationService instance.

    Returns:
        dict[str, Any]: State update dictionary containing 'draft_answer'.
    """
    raw_query = state.get("query", "")
    query = raw_query.strip() if isinstance(raw_query, str) else ""

    if not query:
        logger.warning("Empty query supplied to generate_node.")
        return {"draft_answer": FALLBACK_RESPONSE}

    if state.get("retrieval_relevant") is False:
        logger.info("Retrieval marked not relevant; returning fallback response.")
        return {"draft_answer": FALLBACK_RESPONSE}

    retrieved_context = state.get("retrieved_context", [])
    if not retrieved_context:
        logger.info("No retrieved context available; returning fallback response.")
        return {"draft_answer": FALLBACK_RESPONSE}

    sources = state.get("sources", [])
    service = generator_service or GenerationService()

    try:
        draft_answer = service.generate(
            question=query,
            retrieved_context=retrieved_context,
            sources=sources,
        )
    except Exception as exc:
        logger.error("Failed generating answer for query '%s': %s", query, exc)
        return {"draft_answer": FALLBACK_RESPONSE}

    return {"draft_answer": draft_answer}


def check_grounding_node(
    state: RAGState,
    evaluator: GroundednessEvaluator | None = None,
) -> dict[str, Any]:
    """LangGraph node to verify factual grounding of draft_answer against retrieved_context.

    Args:
        state: Current RAGState dictionary.
        evaluator: Optional GroundednessEvaluator instance.

    Returns:
        dict[str, Any]: State update with 'grounded', 'grounding_score', 'confidence_score'.
    """
    service = evaluator or GroundednessEvaluator()
    query = state.get("query", "")
    draft_answer = state.get("draft_answer", "")
    retrieved_context = state.get("retrieved_context", [])

    if not draft_answer or draft_answer == FALLBACK_RESPONSE:
        return {
            "grounded": False,
            "grounding_score": 0.0,
            "confidence_score": 0.0,
        }

    try:
        result = service.evaluate(
            query=query,
            draft_answer=draft_answer,
            retrieved_context=retrieved_context,
        )
        return {
            "grounded": result.grounded,
            "grounding_score": result.grounding_score,
            "confidence_score": result.grounding_score,
        }
    except Exception as exc:
        logger.error("check_grounding_node failed: %s", exc)
        return {
            "grounded": False,
            "grounding_score": 0.0,
            "confidence_score": 0.0,
        }


def regenerate_node(
    state: RAGState,
    generator_service: GenerationService | None = None,
) -> dict[str, Any]:
    """LangGraph node to regenerate a draft answer with incremented retry count.

    Args:
        state: Current RAGState dictionary.
        generator_service: Optional GenerationService instance.

    Returns:
        dict[str, Any]: State update with 'draft_answer' and 'retry_count'.
    """
    service = generator_service or GenerationService()
    current_retry = int(state.get("retry_count", 0)) + 1
    logger.info("regenerate_node: executing generation retry attempt %d", current_retry)

    query = state.get("query", "")
    retrieved_context = state.get("retrieved_context", [])
    sources = state.get("sources", [])

    try:
        new_draft = service.generate(
            question=query,
            retrieved_context=retrieved_context,
            sources=sources,
        )
    except Exception as exc:
        logger.error("regenerate_node failed: %s", exc)
        new_draft = FALLBACK_RESPONSE

    return {
        "draft_answer": new_draft,
        "retry_count": current_retry,
    }


def decide_grounding(state: RAGState) -> str:
    """Routing condition after check_grounding node.

    Returns:
        str: 'finalize' if grounded is True,
             'regenerate' if not grounded and retry_count < 1,
             'refuse' if not grounded and retry_count >= 1.
    """
    if state.get("grounded") is True:
        return "finalize"
    if int(state.get("retry_count", 0)) < 1:
        return "regenerate"
    return "refuse"


def finalize_node(state: RAGState) -> dict[str, Any]:
    """LangGraph node to finalize the grounded answer.

    Args:
        state: Current RAGState dictionary.

    Returns:
        dict[str, Any]: State update with 'final_answer' and 'confidence_score'.
    """
    draft = state.get("draft_answer") or FALLBACK_RESPONSE
    score = state.get("grounding_score", 1.0)
    logger.info("finalize_node: finalized answer with grounding score %.2f", score)
    return {
        "final_answer": draft,
        "confidence_score": score,
    }


def refuse_node(state: RAGState) -> dict[str, Any]:
    """LangGraph node to set safe refusal response when retrieval or grounding fails.

    Args:
        state: Current RAGState dictionary.

    Returns:
        dict[str, Any]: State update with safe refusal response.
    """
    logger.info("refuse_node: setting fallback refusal response.")
    return {
        "final_answer": FALLBACK_RESPONSE,
        "grounded": False,
        "grounding_score": 0.0,
        "confidence_score": 0.0,
    }
