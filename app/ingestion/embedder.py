"""Embedding service using LangChain's OpenAI embeddings integration."""

from collections.abc import Sequence
import logging
from typing import Any
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

from app.config import get_settings
from app.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Base exception for all embedding failures."""


class MissingAPIKeyError(EmbeddingError):
    """Raised when the OpenAI API key is missing or blank."""


class EmbeddingAPIError(EmbeddingError):
    """Raised when the OpenAI embeddings API call fails."""


class EmbeddedChunk(BaseModel):
    """Document chunk coupled with its vector embedding and preserved metadata."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    source: str = Field(..., description="Source document identifier")
    page: int = Field(..., ge=1, description="Original 1-based page number")
    text: str = Field(..., min_length=1, description="Original text of the chunk")
    embedding: list[float] = Field(..., min_length=1, description="Vector embedding representation")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Preserved metadata dictionary")

    def model_post_init(self, __context: Any) -> None:
        if not self.metadata:
            self.metadata = {
                "chunk_id": self.chunk_id,
                "source": self.source,
                "page": self.page,
            }
        else:
            self.metadata.setdefault("chunk_id", self.chunk_id)
            self.metadata.setdefault("source", self.source)
            self.metadata.setdefault("page", self.page)


class EmbeddingService:
    """Generates OpenAI vector embeddings for document chunks and user queries."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        embeddings_client: OpenAIEmbeddings | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.OPENAI_EMBEDDING_MODEL
        resolved_api_key = api_key or settings.OPENAI_API_KEY

        if not resolved_api_key or not resolved_api_key.strip():
            raise MissingAPIKeyError(
                "OpenAI API key is missing. Set OPENAI_API_KEY in the environment or .env file."
            )

        if embeddings_client is not None:
            self.client = embeddings_client
        else:
            self.client = OpenAIEmbeddings(
                model=self.model,
                api_key=resolved_api_key,
            )

        logger.info("Initialized EmbeddingService with model: %s", self.model)

    def embed_chunks(
        self,
        chunks: Sequence[DocumentChunk | dict[str, Any] | Any],
    ) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        normalized_chunks: list[DocumentChunk] = []
        for chunk in chunks:
            if isinstance(chunk, DocumentChunk):
                normalized_chunks.append(chunk)
            elif isinstance(chunk, dict):
                normalized_chunks.append(DocumentChunk(**chunk))
            else:
                normalized_chunks.append(
                    DocumentChunk(
                        chunk_id=getattr(chunk, "chunk_id", ""),
                        source=getattr(chunk, "source", "unknown"),
                        page=getattr(chunk, "page", 1),
                        text=getattr(chunk, "text", ""),
                        metadata=getattr(chunk, "metadata", {}),
                    )
                )

        texts = [chunk.text for chunk in normalized_chunks]

        try:
            raw_embeddings = self.client.embed_documents(texts)
        except Exception as exc:
            logger.error("Embedding API call failed for model '%s': %s", self.model, exc)
            raise EmbeddingAPIError(f"OpenAI embedding generation failed: {exc}") from exc

        if len(raw_embeddings) != len(normalized_chunks):
            raise EmbeddingAPIError(
                f"Embedding count mismatch: expected {len(normalized_chunks)}, received {len(raw_embeddings)}"
            )

        embedded_chunks: list[EmbeddedChunk] = []
        for chunk, vector in zip(normalized_chunks, raw_embeddings):
            merged_metadata = dict(chunk.metadata)
            merged_metadata.update(
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "page": chunk.page,
                }
            )

            embedded_chunks.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    page=chunk.page,
                    text=chunk.text,
                    embedding=vector,
                    metadata=merged_metadata,
                )
            )

        return embedded_chunks

    def embed_query(self, query: str) -> list[float]:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query string cannot be empty or whitespace.")

        try:
            return self.client.embed_query(clean_query)
        except Exception as exc:
            logger.error("Query embedding API call failed for model '%s': %s", self.model, exc)
            raise EmbeddingAPIError(f"Failed to generate embedding for query: {exc}") from exc


def embed_chunks(
    chunks: Sequence[DocumentChunk | dict[str, Any] | Any],
    model: str | None = None,
    api_key: str | None = None,
    embeddings_client: OpenAIEmbeddings | None = None,
) -> list[EmbeddedChunk]:
    service = EmbeddingService(
        model=model,
        api_key=api_key,
        embeddings_client=embeddings_client,
    )
    return service.embed_chunks(chunks)
