"""API route handlers for health checks and agentic chat."""

import logging
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.generation.generator import GenerationAPIError
from app.graph.workflow import RAGResult, run_rag
from app.retrieval.vector_store import VectorStoreError

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Health Models & Endpoint
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str = Field(default="ok", examples=["ok"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    description="Returns service health status to verify service availability.",
    tags=["Health"],
)
async def get_health() -> HealthResponse:
    """Check health status of the application."""
    return HealthResponse(status="ok")


# ---------------------------------------------------------------------------
# Chat Models & Dependencies
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Schema for incoming chat query requests."""

    query: str = Field(
        ...,
        min_length=1,
        description="User question to be answered strictly from data/Ebook-Agentic-AI.pdf",
        examples=["What is Agentic AI?"],
    )

    @field_validator("query")
    @classmethod
    def validate_query_non_empty(cls, v: str) -> str:
        """Ensure query is not empty or whitespace-only."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Query string cannot be empty or whitespace-only.")
        return v.strip()


class SourceItem(BaseModel):
    """Source citation metadata preserving page and chunk identifiers."""

    page: int = Field(
        ...,
        ge=1,
        description="1-based page number from original PDF document",
        examples=[12],
    )
    chunk_id: str = Field(
        ...,
        description="Unique chunk identifier",
        examples=["page_12_chunk_03"],
    )
    source: str = Field(
        default="Ebook-Agentic-AI.pdf",
        description="Source document file name",
        examples=["Ebook-Agentic-AI.pdf"],
    )


class ChatResponse(BaseModel):
    """Structured response schema for POST /api/v1/chat."""

    query: str = Field(
        ...,
        description="Original query submitted by the user",
        examples=["What is Agentic AI?"],
    )
    final_answer: str = Field(
        ...,
        description="Answer strictly grounded in the document, or refusal fallback",
        examples=[
            "Agentic AI systems operate through autonomous perception, reasoning, and tool action loops."
        ],
    )
    retrieved_context_chunks: list[str] = Field(
        ...,
        description="Text content of the retrieved document chunks",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Deterministic confidence score between 0.0 and 1.0",
        examples=[0.92],
    )
    grounded: bool = Field(
        default=False,
        description="Whether the response is factually grounded in the retrieved context",
    )
    grounding_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Factual grounding score between 0.0 and 1.0",
        examples=[0.95],
    )
    sources: list[SourceItem] = Field(
        default_factory=list,
        description="Source citations with page numbers and chunk identifiers",
    )


def get_rag_pipeline() -> Callable[[str], RAGResult]:
    """Dependency provider returning the RAG execution function."""
    return run_rag


# ---------------------------------------------------------------------------
# Chat Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a document-grounded question",
    description=(
        "Executes the LangGraph Agentic RAG workflow to retrieve relevant context chunks "
        "from Ebook-Agentic-AI.pdf, generates a strictly grounded answer, validates factual "
        "grounding, and calculates a transparent multi-signal confidence score."
    ),
    tags=["Chat"],
)
async def chat_endpoint(
    request: ChatRequest,
    rag_pipeline: Callable[[str], RAGResult] = Depends(get_rag_pipeline),
) -> ChatResponse:
    """Process a user question through the LangGraph RAG pipeline."""
    cleaned_query = request.query.strip()
    if not cleaned_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty or whitespace-only.",
        )

    try:
        rag_result: RAGResult = rag_pipeline(cleaned_query)
    except VectorStoreError as exc:
        logger.error("Vector database error during query '%s': %s", cleaned_query, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector database service is currently unavailable. Please try again later.",
        ) from exc
    except GenerationAPIError as exc:
        logger.error("LLM generation API error during query '%s': %s", cleaned_query, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Language model generation service is currently unavailable. Please try again later.",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected internal error during query '%s': %s", cleaned_query, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing your request.",
        ) from exc

    # Parse and format sources safely
    formatted_sources: list[SourceItem] = []
    for item in rag_result.sources:
        if isinstance(item, dict):
            formatted_sources.append(
                SourceItem(
                    page=int(item.get("page", 1)),
                    chunk_id=str(item.get("chunk_id", "unknown")),
                    source=str(item.get("source", "Ebook-Agentic-AI.pdf")),
                )
            )

    return ChatResponse(
        query=rag_result.query,
        final_answer=rag_result.final_answer,
        retrieved_context_chunks=rag_result.retrieved_context,
        confidence_score=rag_result.confidence_score,
        grounded=rag_result.grounded,
        grounding_score=rag_result.grounding_score,
        sources=formatted_sources,
    )
