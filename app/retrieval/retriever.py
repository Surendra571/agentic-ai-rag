"""Query retrieval service orchestrating embeddings and vector search."""

import logging
from typing import Any
from pydantic import BaseModel, Field

from app.config import get_settings
from app.ingestion.embedder import EmbeddingService
from app.retrieval.vector_store import PineconeVectorStore

logger = logging.getLogger(__name__)


class RetrievedChunk(BaseModel):
    """Structured representation of a retrieved document chunk."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    source: str = Field(..., description="Source document identifier")
    page: int = Field(..., ge=1, description="Original 1-based page number")
    text: str = Field(..., min_length=1, description="Extracted textual content of the chunk")
    similarity_score: float = Field(..., description="Similarity score between query and chunk")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Preserved metadata dictionary")


class RetrievalResponse(BaseModel):
    """Complete structured response from the retrieval pipeline."""

    query: str = Field(..., description="Original user question")
    chunks: list[RetrievedChunk] = Field(default_factory=list, description="Retrieved chunks")
    top_k: int = Field(..., description="Configured top_k threshold")
    similarity_threshold: float = Field(..., description="Minimum similarity threshold applied")


class RetrieverService:
    """Orchestrates query validation, embedding generation, Pinecone retrieval, and threshold filtering."""

    def __init__(
        self,
        vector_store: PineconeVectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> None:
        """Initialize RetrieverService with dependencies and configuration.

        Args:
            vector_store: Vector store instance for similarity search.
            embedding_service: Embedding service for query vectorization.
            top_k: Target number of chunks to retrieve. Defaults to settings.RETRIEVAL_TOP_K (5).
            similarity_threshold: Minimum similarity score required. Defaults to settings.MINIMUM_SIMILARITY_THRESHOLD.
        """
        settings = get_settings()
        self.top_k = top_k if top_k is not None else settings.RETRIEVAL_TOP_K
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.MINIMUM_SIMILARITY_THRESHOLD
        )

        self.vector_store = vector_store or PineconeVectorStore()
        self.embedding_service = embedding_service or EmbeddingService()

    def validate_query(self, query: str) -> str:
        """Validate that the query is a non-empty string.

        Args:
            query: Input user query.

        Returns:
            str: Stripped query string.

        Raises:
            ValueError: If query is not a string, empty, or whitespace-only.
        """
        if not isinstance(query, str):
            raise ValueError(f"Query must be a string, got {type(query).__name__}")

        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Query string cannot be empty or whitespace-only.")

        return cleaned

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
        namespace: str | None = None,
    ) -> list[RetrievedChunk]:
        """Execute semantic retrieval pipeline for the user query.

        Steps:
        1. Validate query.
        2. Generate query embedding.
        3. Search Pinecone for top-k candidates.
        4. Filter out matches below the minimum similarity threshold.
        5. Return structured results without hallucinating context.

        Args:
            query: User's search question.
            top_k: Optional override for number of chunks to fetch.
            threshold: Optional override for minimum similarity score.
            namespace: Optional Pinecone namespace override.

        Returns:
            list[RetrievedChunk]: Filtered list of relevant chunks, or empty list if none qualify.
        """
        clean_query = self.validate_query(query)
        effective_top_k = top_k if top_k is not None else self.top_k
        effective_threshold = (
            threshold if threshold is not None else self.similarity_threshold
        )

        if effective_top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        # 1. Generate query embedding
        query_vector = self.embedding_service.embed_query(clean_query)

        # 2. Search Pinecone
        raw_results = self.vector_store.similarity_search_by_vector(
            query_vector=query_vector,
            top_k=effective_top_k,
            namespace=namespace,
        )

        # 3. Filter low-quality matches and preserve metadata
        relevant_chunks: list[RetrievedChunk] = []
        for result in raw_results:
            if result.similarity_score < effective_threshold:
                logger.debug(
                    "Filtered chunk %s: similarity score %.4f < threshold %.4f",
                    result.chunk_id,
                    result.similarity_score,
                    effective_threshold,
                )
                continue

            relevant_chunks.append(
                RetrievedChunk(
                    chunk_id=result.chunk_id,
                    source=result.metadata.get("source", "Ebook-Agentic-AI.pdf"),
                    page=result.page,
                    text=result.text,
                    similarity_score=result.similarity_score,
                    metadata=result.metadata,
                )
            )

        return relevant_chunks

    def retrieve_with_details(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
        namespace: str | None = None,
    ) -> RetrievalResponse:
        """Execute retrieval and return full details including query and parameters."""
        clean_query = self.validate_query(query)
        effective_top_k = top_k if top_k is not None else self.top_k
        effective_threshold = (
            threshold if threshold is not None else self.similarity_threshold
        )

        chunks = self.retrieve(
            query=clean_query,
            top_k=effective_top_k,
            threshold=effective_threshold,
            namespace=namespace,
        )

        return RetrievalResponse(
            query=clean_query,
            chunks=chunks,
            top_k=effective_top_k,
            similarity_threshold=effective_threshold,
        )
