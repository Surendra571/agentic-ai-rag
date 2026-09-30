"""LangGraph Agentic RAG workflow assembly and execution service."""

import logging
from typing import Any
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.generation.generator import FALLBACK_RESPONSE, GenerationService
from app.grading.groundedness import GroundednessEvaluator
from app.graph.nodes import (
    check_grounding_node,
    decide_grounding,
    decide_retrieval,
    finalize_node,
    generate_node,
    grade_retrieval_node,
    refuse_node,
    regenerate_node,
    retrieve_node,
)
from app.graph.state import RAGState, create_initial_state
from app.retrieval.retriever import RetrieverService

logger = logging.getLogger(__name__)


class RAGResult(BaseModel):
    """Structured result returned by run_rag execution."""

    query: str = Field(..., description="Original user query")
    final_answer: str = Field(..., description="Final grounded answer or safe refusal")
    grounded: bool = Field(..., description="Whether final answer is supported by context")
    grounding_score: float = Field(..., ge=0.0, le=1.0, description="Grounding score between 0.0 and 1.0")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    retrieval_relevant: bool = Field(..., description="Whether retrieved context was relevant")
    sources: list[dict[str, Any]] = Field(default_factory=list, description="Citations / sources")
    retrieved_context: list[str] = Field(default_factory=list, description="Retrieved context chunks")
    retry_count: int = Field(default=0, ge=0, description="Number of generation retries performed")

    def to_dict(self) -> dict[str, Any]:
        """Return clean dictionary representation."""
        return self.model_dump()


def build_rag_workflow(
    retriever_service: RetrieverService | None = None,
    generator_service: GenerationService | None = None,
    groundedness_evaluator: GroundednessEvaluator | None = None,
) -> Any:
    """Construct and compile the LangGraph Agentic RAG StateGraph.

    Graph topology:
    START
      ↓
    retrieve
      ↓
    grade_retrieval
      ├── insufficient → refuse → END
      ↓
    generate
      ↓
    check_grounding
      ├── grounded → finalize → END
      ├── not grounded (<1 retry) → regenerate → check_grounding
      └── still unsupported (>=1 retry) → refuse → END

    Args:
        retriever_service: Optional RetrieverService instance override.
        generator_service: Optional GenerationService instance override.
        groundedness_evaluator: Optional GroundednessEvaluator instance override.

    Returns:
        CompiledStateGraph: Compiled runnable LangGraph workflow.
    """
    workflow = StateGraph(RAGState)

    # 1. Define nodes with injected dependencies
    def _retrieve(state: RAGState) -> dict[str, Any]:
        return retrieve_node(state, retriever_service=retriever_service)

    def _grade_retrieval(state: RAGState) -> dict[str, Any]:
        return grade_retrieval_node(state)

    def _generate(state: RAGState) -> dict[str, Any]:
        return generate_node(state, generator_service=generator_service)

    def _check_grounding(state: RAGState) -> dict[str, Any]:
        return check_grounding_node(state, evaluator=groundedness_evaluator)

    def _regenerate(state: RAGState) -> dict[str, Any]:
        return regenerate_node(state, generator_service=generator_service)

    def _finalize(state: RAGState) -> dict[str, Any]:
        return finalize_node(state)

    def _refuse(state: RAGState) -> dict[str, Any]:
        return refuse_node(state)

    workflow.add_node("retrieve", _retrieve)
    workflow.add_node("grade_retrieval", _grade_retrieval)
    workflow.add_node("generate", _generate)
    workflow.add_node("check_grounding", _check_grounding)
    workflow.add_node("regenerate", _regenerate)
    workflow.add_node("finalize", _finalize)
    workflow.add_node("refuse", _refuse)

    # 2. Define edges & conditional transitions
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_retrieval")

    workflow.add_conditional_edges(
        "grade_retrieval",
        decide_retrieval,
        {
            "generate": "generate",
            "refuse": "refuse",
        },
    )

    workflow.add_edge("generate", "check_grounding")

    workflow.add_conditional_edges(
        "check_grounding",
        decide_grounding,
        {
            "finalize": "finalize",
            "regenerate": "regenerate",
            "refuse": "refuse",
        },
    )

    # Max 1 retry: regenerate routes back to check_grounding
    workflow.add_edge("regenerate", "check_grounding")

    workflow.add_edge("finalize", END)
    workflow.add_edge("refuse", END)

    return workflow.compile()


def run_rag(
    query: str,
    workflow: Any | None = None,
    retriever_service: RetrieverService | None = None,
    generator_service: GenerationService | None = None,
    groundedness_evaluator: GroundednessEvaluator | None = None,
) -> RAGResult:
    """Execute the complete document-grounded Agentic RAG workflow for a user query.

    Args:
        query: User input question.
        workflow: Optional pre-compiled LangGraph runnable (defaults to built workflow).
        retriever_service: Optional RetrieverService instance override.
        generator_service: Optional GenerationService instance override.
        groundedness_evaluator: Optional GroundednessEvaluator instance override.

    Returns:
        RAGResult: Structured response containing final answer, grounding status,
                   confidence score, retrieval status, and citation metadata.

    Raises:
        ValueError: If query is invalid or empty.
    """
    initial_state = create_initial_state(query)

    app = workflow or build_rag_workflow(
        retriever_service=retriever_service,
        generator_service=generator_service,
        groundedness_evaluator=groundedness_evaluator,
    )

    final_state = app.invoke(initial_state)

    final_answer = final_state.get("final_answer") or FALLBACK_RESPONSE
    grounded = bool(final_state.get("grounded", False))
    grounding_score = float(final_state.get("grounding_score") or 0.0)
    confidence_score = float(final_state.get("confidence_score") or 0.0)
    retrieval_relevant = bool(final_state.get("retrieval_relevant", False))
    sources = list(final_state.get("sources", []))
    retrieved_context = list(final_state.get("retrieved_context", []))
    retry_count = int(final_state.get("retry_count", 0))

    return RAGResult(
        query=initial_state["query"],
        final_answer=final_answer,
        grounded=grounded,
        grounding_score=grounding_score,
        confidence_score=confidence_score,
        retrieval_relevant=retrieval_relevant,
        sources=sources,
        retrieved_context=retrieved_context,
        retry_count=retry_count,
    )


def generate_mermaid_graph() -> str:
    """Generate Mermaid syntax diagram of the compiled Agentic RAG StateGraph."""
    try:
        app = build_rag_workflow(
            retriever_service=RetrieverService.__new__(RetrieverService),
            generator_service=GenerationService.__new__(GenerationService),
            groundedness_evaluator=GroundednessEvaluator.__new__(GroundednessEvaluator),
        )
        return app.get_graph().draw_mermaid()
    except Exception as exc:
        logger.warning("Could not generate mermaid diagram via LangGraph: %s", exc)
        return """flowchart TD
    START --> retrieve
    retrieve --> grade_retrieval
    grade_retrieval -->|insufficient| refuse
    grade_retrieval -->|relevant| generate
    generate --> check_grounding
    check_grounding -->|grounded| finalize
    check_grounding -->|not grounded retry < 1| regenerate
    check_grounding -->|still unsupported retry >= 1| refuse
    regenerate --> check_grounding
    finalize --> END
    refuse --> END"""
