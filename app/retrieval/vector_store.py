"""Pinecone vector store integration and retrieval service."""

from collections.abc import Sequence
import logging
from typing import Any
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, Field

from app.config import get_settings
from app.ingestion.embedder import EmbeddedChunk

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Base exception for all vector store errors."""


class PineconeConnectionError(VectorStoreError):
    """Raised when connecting to or initializing Pinecone fails."""


class IndexNotFoundError(VectorStoreError):
    """Raised when the specified index does not exist and cannot be created."""


class VectorStoreUpsertError(VectorStoreError):
    """Raised when upserting vectors into Pinecone fails."""


class VectorStoreQueryError(VectorStoreError):
    """Raised when querying vectors from Pinecone fails."""


class SearchResult(BaseModel):
    """Structured representation of a vector similarity retrieval result."""

    text: str = Field(..., description="Retrieved chunk text")
    page: int = Field(..., ge=1, description="Source page number")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    similarity_score: float = Field(..., description="Similarity score (e.g. cosine)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")


class PineconeVectorStore:
    """Production vector store implementation backed by Pinecone."""

    DEFAULT_DIMENSION: int = 1536
    DEFAULT_TOP_K: int = 5

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
        namespace: str | None = None,
        client: Any | None = None,
        index: Any | None = None,
    ) -> None:
        """Initialize Pinecone vector store with credentials and target index.

        Args:
            api_key: Pinecone API key. Defaults to settings.PINECONE_API_KEY.
            index_name: Pinecone index name. Defaults to settings.PINECONE_INDEX_NAME.
            namespace: Pinecone namespace partition. Defaults to settings.PINECONE_NAMESPACE.
            client: Optional mock or pre-configured Pinecone client.
            index: Optional mock or pre-configured Pinecone Index instance.
        """
        settings = get_settings()
        self.api_key = api_key or settings.PINECONE_API_KEY
        self.index_name = index_name or settings.PINECONE_INDEX_NAME
        self.namespace = namespace or settings.PINECONE_NAMESPACE

        if not self.api_key or not self.api_key.strip():
            raise PineconeConnectionError(
                "Pinecone API key is missing. Set PINECONE_API_KEY in the environment or .env file."
            )
        if not self.index_name or not self.index_name.strip():
            raise PineconeConnectionError("Pinecone index name cannot be blank.")

        self._client = client
        self._index = index

    @property
    def client(self) -> Pinecone:
        """Lazily initialize and return the Pinecone client."""
        if self._client is None:
            try:
                self._client = Pinecone(api_key=self.api_key)
            except Exception as exc:
                raise PineconeConnectionError(f"Failed to initialize Pinecone client: {exc}") from exc
        return self._client

    def ensure_index(
        self,
        dimension: int = DEFAULT_DIMENSION,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        create_if_missing: bool = True,
    ) -> Any:
        """Verify the index exists, creating it if appropriate.

        Args:
            dimension: Vector dimensionality.
            metric: Similarity metric (cosine, euclidean, dotproduct).
            cloud: Serverless cloud provider.
            region: Serverless region.
            create_if_missing: Whether to automatically create the index if it does not exist.

        Returns:
            The connected Pinecone Index instance.
        """
        if self._index is not None:
            return self._index

        client = self.client

        try:
            index_exists = client.has_index(self.index_name)
        except Exception as exc:
            raise PineconeConnectionError(f"Failed to check Pinecone index existence: {exc}") from exc

        if not index_exists:
            if not create_if_missing:
                raise IndexNotFoundError(f"Pinecone index '{self.index_name}' does not exist.")

            try:
                logger.info(
                    "Creating Pinecone index '%s' (dim=%d, metric=%s)...",
                    self.index_name,
                    dimension,
                    metric,
                )
                client.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(cloud=cloud, region=region),
                )
            except Exception as exc:
                raise PineconeConnectionError(
                    f"Failed to create Pinecone index '{self.index_name}': {exc}"
                ) from exc

        try:
            self._index = client.Index(self.index_name)
            return self._index
        except Exception as exc:
            raise PineconeConnectionError(
                f"Failed to connect to Pinecone index '{self.index_name}': {exc}"
            ) from exc

    def get_index(self) -> Any:
        """Get the active index connection, ensuring it exists."""
        if self._index is None:
            return self.ensure_index()
        return self._index

    def upsert_chunks(
        self,
        chunks: Sequence[EmbeddedChunk | dict[str, Any] | Any],
        batch_size: int = 100,
        namespace: str | None = None,
    ) -> int:
        """Upload embedded document chunks into Pinecone with preserved metadata.

        Args:
            chunks: Sequence of EmbeddedChunk instances or matching dictionaries.
            batch_size: Number of vectors per upsert batch.
            namespace: Target namespace (defaults to configured namespace).

        Returns:
            int: Total number of vectors successfully upserted.

        Raises:
            VectorStoreUpsertError: If the upsert operation fails.
        """
        if not chunks:
            return 0

        target_namespace = namespace or self.namespace
        index = self.get_index()

        vectors_to_upsert: list[dict[str, Any]] = []
        for chunk in chunks:
            if isinstance(chunk, EmbeddedChunk):
                chunk_id = chunk.chunk_id
                embedding = chunk.embedding
                text = chunk.text
                source = chunk.source
                page = chunk.page
                meta = dict(chunk.metadata)
            elif isinstance(chunk, dict):
                chunk_id = chunk["chunk_id"]
                embedding = chunk["embedding"]
                text = chunk["text"]
                source = chunk["source"]
                page = chunk["page"]
                meta = dict(chunk.get("metadata", {}))
            else:
                chunk_id = str(getattr(chunk, "chunk_id"))
                embedding = list(getattr(chunk, "embedding"))
                text = str(getattr(chunk, "text"))
                source = str(getattr(chunk, "source"))
                page = int(getattr(chunk, "page"))
                meta = dict(getattr(chunk, "metadata", {}))

            # Store required metadata fields
            meta["text"] = text
            meta["source"] = source
            meta["page"] = page
            meta["chunk_id"] = chunk_id

            vectors_to_upsert.append(
                {
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": meta,
                }
            )

        total_upserted = 0
        try:
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i : i + batch_size]
                index.upsert(vectors=batch, namespace=target_namespace)
                total_upserted += len(batch)
        except Exception as exc:
            raise VectorStoreUpsertError(f"Failed to upsert vectors into Pinecone: {exc}") from exc

        return total_upserted

    def similarity_search_by_vector(
        self,
        query_vector: list[float],
        top_k: int = DEFAULT_TOP_K,
        namespace: str | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Query Pinecone using an embedding vector and return structured SearchResults.

        Args:
            query_vector: Float list representing query embedding.
            top_k: Number of most similar vectors to retrieve (default 5).
            namespace: Target namespace (defaults to configured namespace).
            filter_metadata: Optional Pinecone metadata filter.

        Returns:
            list[SearchResult]: Ranked list of search results.

        Raises:
            ValueError: If query_vector is empty or top_k <= 0.
            VectorStoreQueryError: If the query fails.
        """
        if not query_vector:
            raise ValueError("Query vector cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        target_namespace = namespace or self.namespace
        index = self.get_index()

        query_kwargs: dict[str, Any] = {
            "vector": query_vector,
            "top_k": top_k,
            "namespace": target_namespace,
            "include_metadata": True,
            "include_values": False,
        }
        if filter_metadata:
            query_kwargs["filter"] = filter_metadata

        try:
            response = index.query(**query_kwargs)
        except Exception as exc:
            raise VectorStoreQueryError(f"Pinecone similarity query failed: {exc}") from exc

        results: list[SearchResult] = []
        matches = response.matches if hasattr(response, "matches") else response.get("matches", [])

        for match in matches:
            match_id = match.id if hasattr(match, "id") else match.get("id", "")
            score = float(match.score if hasattr(match, "score") else match.get("score", 0.0))
            metadata = dict(match.metadata if hasattr(match, "metadata") else match.get("metadata", {}))

            text = metadata.get("text", "")
            page = int(metadata.get("page", 1))
            chunk_id = metadata.get("chunk_id", match_id)

            results.append(
                SearchResult(
                    text=text,
                    page=page,
                    chunk_id=chunk_id,
                    similarity_score=score,
                    metadata=metadata,
                )
            )

        return results

    def similarity_search(
        self,
        query: str,
        embedding_service: Any,
        top_k: int = DEFAULT_TOP_K,
        namespace: str | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Convenience method to embed query string and perform similarity search.

        Args:
            query: User question or search query string.
            embedding_service: Instance of EmbeddingService with embed_query method.
            top_k: Number of results to return.
            namespace: Target namespace.
            filter_metadata: Optional metadata filter.

        Returns:
            list[SearchResult]: Ranked search results.
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty or whitespace.")

        query_vector = embedding_service.embed_query(query)

        return self.similarity_search_by_vector(
            query_vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            filter_metadata=filter_metadata,
        )
