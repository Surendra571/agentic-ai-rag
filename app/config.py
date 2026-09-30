"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Application Configuration
    APP_NAME: str = "Agentic AI RAG API"
    APP_DESCRIPTION: str = "Document-grounded Agentic AI RAG chatbot service"
    APP_VERSION: str = "0.1.0"
    APP_ENV: Literal["development", "staging", "production", "test"] = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # OpenAI Configuration
    OPENAI_API_KEY: str = Field(
        default="",
        description="OpenAI API key",
    )
    OPENAI_EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model identifier",
    )
    OPENAI_CHAT_MODEL: str = Field(
        default="gpt-4o-mini",
        description="OpenAI chat completion model identifier",
    )

    # Pinecone Configuration
    PINECONE_API_KEY: str = Field(
        default="",
        description="Pinecone API key",
    )
    PINECONE_INDEX_NAME: str = Field(
        default="agentic-ai-rag",
        description="Pinecone index name for document embeddings",
    )
    PINECONE_NAMESPACE: str = Field(
        default="default",
        description="Pinecone namespace for document vectors",
    )

    # Document Chunking Configuration
    CHUNK_SIZE: int = Field(
        default=800,
        ge=50,
        description="Default character size for text chunks",
    )
    CHUNK_OVERLAP: int = Field(
        default=100,
        ge=0,
        description="Character overlap between consecutive chunks",
    )

    # Retrieval Configuration
    RETRIEVAL_TOP_K: int = Field(
        default=5,
        ge=1,
        description="Default number of top similar chunks to retrieve",
    )
    MINIMUM_SIMILARITY_THRESHOLD: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score required to keep a chunk",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of the application settings."""
    return Settings()
