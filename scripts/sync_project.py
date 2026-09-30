"""Script to synchronize and ensure all project files are fully populated."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

FILES: dict[str, str] = {}

FILES[".gitignore"] = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environments
.venv/
venv/
env/
ENV/

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.pytest_cache/

# Environment variables
.env
.env.local

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Data files (keep .gitkeep)
data/*.pdf
!data/.gitkeep
"""

FILES["requirements.txt"] = """fastapi>=0.115.0
uvicorn[standard]>=0.32.0
pydantic>=2.9.0
pydantic-settings>=2.5.0
python-dotenv>=1.0.1
pymupdf>=1.24.10
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-pinecone>=0.2.0
langgraph>=0.2.20
pinecone>=5.3.0
langchain-text-splitters>=0.3.0
pytest>=8.3.0
httpx>=0.27.0
"""

FILES[".env.example"] = """# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=agentic-ai-rag
PINECONE_NAMESPACE=default

# Document Chunking Configuration
CHUNK_SIZE=800
CHUNK_OVERLAP=100

# Application Configuration
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
"""

FILES["app/config.py"] = """\"\"\"Application configuration using Pydantic Settings.\"\"\"

from functools import lru_cache
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    \"\"\"Application settings loaded from environment variables and .env file.\"\"\"

    model_config = SettingsConfigDict(
        env_file=\".env\",
        env_file_encoding=\"utf-8\",
        case_sensitive=True,
        extra=\"ignore\",
    )

    # Application Configuration
    APP_NAME: str = \"Agentic AI RAG API\"
    APP_DESCRIPTION: str = \"Document-grounded Agentic AI RAG chatbot service\"
    APP_VERSION: str = \"0.1.0\"
    APP_ENV: Literal[\"development\", \"staging\", \"production\", \"test\"] = \"development\"
    APP_HOST: str = \"0.0.0.0\"
    APP_PORT: int = 8000
    LOG_LEVEL: str = \"INFO\"

    # OpenAI Configuration
    OPENAI_API_KEY: str = Field(..., description=\"OpenAI API key\")
    OPENAI_EMBEDDING_MODEL: str = Field(
        default=\"text-embedding-3-small\",
        description=\"OpenAI embedding model identifier\",
    )
    OPENAI_CHAT_MODEL: str = Field(
        default=\"gpt-4o-mini\",
        description=\"OpenAI chat completion model identifier\",
    )

    # Pinecone Configuration
    PINECONE_API_KEY: str = Field(..., description=\"Pinecone API key\")
    PINECONE_INDEX_NAME: str = Field(
        ...,
        description=\"Pinecone index name for document embeddings\",
    )
    PINECONE_NAMESPACE: str = Field(
        default=\"default\",
        description=\"Pinecone namespace for document vectors\",
    )

    # Document Chunking Configuration
    CHUNK_SIZE: int = Field(
        default=800,
        ge=50,
        description=\"Default character size for text chunks\",
    )
    CHUNK_OVERLAP: int = Field(
        default=100,
        ge=0,
        description=\"Character overlap between consecutive chunks\",
    )


@lru_cache
def get_settings() -> Settings:
    \"\"\"Return a cached instance of the application settings.\"\"\"
    return Settings()
"""

FILES["app/ingestion/loader.py"] = """\"\"\"PDF document loader and extractor module using PyMuPDF.\"\"\"

from pathlib import Path
from typing import Any
import pymupdf
from pydantic import BaseModel, Field


class PDFIngestionError(Exception):
    \"\"\"Base exception for all PDF ingestion errors.\"\"\"


class PDFNotFoundError(PDFIngestionError, FileNotFoundError):
    \"\"\"Raised when the specified PDF file cannot be found.\"\"\"


class PDFReadError(PDFIngestionError):
    \"\"\"Raised when the PDF file cannot be read, opened, or is corrupt.\"\"\"


class EmptyPDFError(PDFIngestionError):
    \"\"\"Raised when a PDF file has no pages or contains no extractable text.\"\"\"


class DocumentPage(BaseModel):
    \"\"\"Structured representation of an extracted document page.\"\"\"

    text: str = Field(..., min_length=1, description=\"Extracted textual content of the page\")
    page: int = Field(..., ge=1, description=\"1-based page number\")
    source: str = Field(..., description=\"Source document filename or identifier\")
    metadata: dict[str, Any] = Field(default_factory=dict, description=\"Metadata dictionary\")

    def model_post_init(self, __context: Any) -> None:
        \"\"\"Populate default metadata if not explicitly provided.\"\"\"
        if not self.metadata:
            self.metadata = {
                \"source\": self.source,
                \"page\": self.page,
            }


class PDFLoader:
    \"\"\"Safely loads and extracts text page-by-page from PDF documents using PyMuPDF.\"\"\"

    def __init__(self, file_path: str | Path = \"data/Ebook-Agentic-AI.pdf\") -> None:
        self.file_path = Path(file_path)

    def load(self) -> list[DocumentPage]:
        if not self.file_path.exists():
            raise PDFNotFoundError(f\"PDF file not found at: {self.file_path}\")

        if not self.file_path.is_file():
            raise PDFNotFoundError(f\"Path is not a regular file: {self.file_path}\")

        try:
            doc = pymupdf.open(self.file_path)
        except Exception as exc:
            raise PDFReadError(f\"Failed to open PDF file '{self.file_path}': {exc}\") from exc

        try:
            if len(doc) == 0:
                raise EmptyPDFError(f\"PDF file '{self.file_path.name}' contains 0 pages.\")

            source_name = self.file_path.name
            pages: list[DocumentPage] = []

            for page_index in range(len(doc)):
                page = doc[page_index]
                try:
                    text = page.get_text()
                except Exception as exc:
                    raise PDFReadError(
                        f\"Failed to extract text from page {page_index + 1} of '{self.file_path.name}': {exc}\"
                    ) from exc

                clean_text = text.strip()

                if not clean_text:
                    continue

                page_number = page_index + 1
                pages.append(
                    DocumentPage(
                        text=clean_text,
                        page=page_number,
                        source=source_name,
                        metadata={
                            \"source\": source_name,
                            \"page\": page_number,
                        },
                    )
                )

            if not pages:
                raise EmptyPDFError(
                    f\"PDF file '{self.file_path.name}' has {len(doc)} pages, but contains no extractable text.\"
                )

            return pages
        finally:
            doc.close()


def load_pdf(file_path: str | Path = \"data/Ebook-Agentic-AI.pdf\") -> list[DocumentPage]:
    loader = PDFLoader(file_path=file_path)
    return loader.load()
"""

FILES["app/ingestion/chunker.py"] = """\"\"\"Document chunking module for vector retrieval using LangChain text splitting.\"\"\"

from collections.abc import Sequence
from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from app.config import get_settings
from app.ingestion.loader import DocumentPage


class DocumentChunk(BaseModel):
    \"\"\"Structured representation of a document chunk.\"\"\"

    chunk_id: str = Field(..., description=\"Unique chunk identifier (e.g. page_12_chunk_03)\")
    source: str = Field(..., description=\"Source document identifier or filename\")
    page: int = Field(..., ge=1, description=\"Original 1-based page number\")
    text: str = Field(..., min_length=1, description=\"Textual content of the chunk\")
    metadata: dict[str, Any] = Field(default_factory=dict, description=\"Metadata dictionary\")

    def model_post_init(self, __context: Any) -> None:
        if not self.metadata:
            self.metadata = {
                \"chunk_id\": self.chunk_id,
                \"source\": self.source,
                \"page\": self.page,
            }


class DocumentChunker:
    \"\"\"Chunks DocumentPage instances into structured DocumentChunk objects.\"\"\"

    DEFAULT_SEPARATORS: list[str] = [
        \"\\n\\n\",
        \"\\n\",
        \" \",
        \"\",
    ]

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ) -> None:
        settings = get_settings()
        self.chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
        self.separators = separators or self.DEFAULT_SEPARATORS

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f\"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})\"
            )

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            keep_separator=True,
            strip_whitespace=True,
        )

    def split_page(self, page: DocumentPage | dict[str, Any] | Any) -> list[DocumentChunk]:
        if isinstance(page, DocumentPage):
            raw_text = page.text
            page_num = page.page
            source = page.source
        elif isinstance(page, dict):
            raw_text = str(page.get(\"text\", \"\"))
            page_num = int(page.get(\"page\", 1))
            source = str(page.get(\"source\", \"unknown\"))
        else:
            raw_text = str(getattr(page, \"text\", \"\"))
            page_num = int(getattr(page, \"page\", 1))
            source = str(getattr(page, \"source\", \"unknown\"))

        clean_text = raw_text.strip()
        if not clean_text:
            return []

        text_pieces = self._splitter.split_text(clean_text)
        chunks: list[DocumentChunk] = []

        chunk_idx = 1
        for piece in text_pieces:
            stripped = piece.strip()
            if not stripped:
                continue

            chunk_id = f\"page_{page_num:02d}_chunk_{chunk_idx:02d}\"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source=source,
                    page=page_num,
                    text=stripped,
                    metadata={
                        \"chunk_id\": chunk_id,
                        \"source\": source,
                        \"page\": page_num,
                    },
                )
            )
            chunk_idx += 1

        return chunks

    def chunk_documents(
        self,
        documents: Sequence[DocumentPage | dict[str, Any] | Any],
    ) -> list[DocumentChunk]:
        all_chunks: list[DocumentChunk] = []
        seen_ids: set[str] = set()

        for doc in documents:
            page_chunks = self.split_page(doc)
            for chunk in page_chunks:
                final_id = chunk.chunk_id
                if final_id in seen_ids:
                    counter = 2
                    while f\"{final_id}_{counter:02d}\" in seen_ids:
                        counter += 1
                    final_id = f\"{final_id}_{counter:02d}\"
                    chunk.chunk_id = final_id
                    chunk.metadata[\"chunk_id\"] = final_id

                seen_ids.add(final_id)
                all_chunks.append(chunk)

        return all_chunks


def chunk_documents(
    documents: Sequence[DocumentPage | dict[str, Any] | Any],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
) -> list[DocumentChunk]:
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    return chunker.chunk_documents(documents)
"""

FILES["app/ingestion/embedder.py"] = """\"\"\"Embedding service using LangChain's OpenAI embeddings integration.\"\"\"

from collections.abc import Sequence
import logging
from typing import Any
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

from app.config import get_settings
from app.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    \"\"\"Base exception for all embedding failures.\"\"\"


class MissingAPIKeyError(EmbeddingError):
    \"\"\"Raised when the OpenAI API key is missing or blank.\"\"\"


class EmbeddingAPIError(EmbeddingError):
    \"\"\"Raised when the OpenAI embeddings API call fails.\"\"\"


class EmbeddedChunk(BaseModel):
    \"\"\"Document chunk coupled with its vector embedding and preserved metadata.\"\"\"

    chunk_id: str = Field(..., description=\"Unique chunk identifier\")
    source: str = Field(..., description=\"Source document identifier\")
    page: int = Field(..., ge=1, description=\"Original 1-based page number\")
    text: str = Field(..., min_length=1, description=\"Original text of the chunk\")
    embedding: list[float] = Field(..., min_length=1, description=\"Vector embedding representation\")
    metadata: dict[str, Any] = Field(default_factory=dict, description=\"Preserved metadata dictionary\")

    def model_post_init(self, __context: Any) -> None:
        if not self.metadata:
            self.metadata = {
                \"chunk_id\": self.chunk_id,
                \"source\": self.source,
                \"page\": self.page,
            }
        else:
            self.metadata.setdefault(\"chunk_id\", self.chunk_id)
            self.metadata.setdefault(\"source\", self.source)
            self.metadata.setdefault(\"page\", self.page)


class EmbeddingService:
    \"\"\"Generates OpenAI vector embeddings for document chunks and user queries.\"\"\"

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
                \"OpenAI API key is missing. Set OPENAI_API_KEY in the environment or .env file.\"
            )

        if embeddings_client is not None:
            self.client = embeddings_client
        else:
            self.client = OpenAIEmbeddings(
                model=self.model,
                api_key=resolved_api_key,
            )

        logger.info(\"Initialized EmbeddingService with model: %s\", self.model)

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
                        chunk_id=getattr(chunk, \"chunk_id\", \"\"),
                        source=getattr(chunk, \"source\", \"unknown\"),
                        page=getattr(chunk, \"page\", 1),
                        text=getattr(chunk, \"text\", \"\"),
                        metadata=getattr(chunk, \"metadata\", {}),
                    )
                )

        texts = [chunk.text for chunk in normalized_chunks]

        try:
            raw_embeddings = self.client.embed_documents(texts)
        except Exception as exc:
            logger.error(\"Embedding API call failed for model '%s': %s\", self.model, exc)
            raise EmbeddingAPIError(f\"OpenAI embedding generation failed: {exc}\") from exc

        if len(raw_embeddings) != len(normalized_chunks):
            raise EmbeddingAPIError(
                f\"Embedding count mismatch: expected {len(normalized_chunks)}, received {len(raw_embeddings)}\"
            )

        embedded_chunks: list[EmbeddedChunk] = []
        for chunk, vector in zip(normalized_chunks, raw_embeddings):
            merged_metadata = dict(chunk.metadata)
            merged_metadata.update(
                {
                    \"chunk_id\": chunk.chunk_id,
                    \"source\": chunk.source,
                    \"page\": chunk.page,
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
            raise ValueError(\"Query string cannot be empty or whitespace.\")

        try:
            return self.client.embed_query(clean_query)
        except Exception as exc:
            logger.error(\"Query embedding API call failed for model '%s': %s\", self.model, exc)
            raise EmbeddingAPIError(f\"Failed to generate embedding for query: {exc}\") from exc


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
"""

FILES["app/ingestion/__init__.py"] = """\"\"\"Document ingestion and parsing module.\"\"\"

from app.ingestion.chunker import (
    DocumentChunk,
    DocumentChunker,
    chunk_documents,
)
from app.ingestion.embedder import (
    EmbeddedChunk,
    EmbeddingAPIError,
    EmbeddingError,
    EmbeddingService,
    MissingAPIKeyError,
    embed_chunks,
)
from app.ingestion.loader import (
    DocumentPage,
    EmptyPDFError,
    PDFIngestionError,
    PDFLoader,
    PDFNotFoundError,
    PDFReadError,
    load_pdf,
)

__all__ = [
    \"DocumentChunk\",
    \"DocumentChunker\",
    \"DocumentPage\",
    \"EmbeddedChunk\",
    \"EmbeddingAPIError\",
    \"EmbeddingError\",
    \"EmbeddingService\",
    \"EmptyPDFError\",
    \"MissingAPIKeyError\",
    \"PDFIngestionError\",
    \"PDFLoader\",
    \"PDFNotFoundError\",
    \"PDFReadError\",
    \"chunk_documents\",
    \"embed_chunks\",
    \"load_pdf\",
]
"""

FILES["tests/test_ingestion.py"] = """\"\"\"Tests for PDF ingestion layer.\"\"\"

from pathlib import Path
import pymupdf
import pytest

from app.ingestion.loader import (
    DocumentPage,
    EmptyPDFError,
    PDFLoader,
    PDFNotFoundError,
    PDFReadError,
    load_pdf,
)


@pytest.fixture
def multi_page_pdf_with_blanks(tmp_path: Path) -> Path:
    pdf_path = tmp_path / \"sample_multipage.pdf\"
    doc = pymupdf.open()

    p1 = doc.new_page()
    p1.insert_text((72, 72), \"Chapter 1: Foundations of Agentic AI Systems.\")
    doc.new_page()
    p3 = doc.new_page()
    p3.insert_text((72, 72), \"Chapter 2: Retrieval-Augmented Generation Workflows.\")

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def whitespace_only_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / \"whitespace_only.pdf\"
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), \"   \\n\\t  \\n  \")
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def zero_page_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / \"zero_pages.pdf\"
    minimal_zero_page_pdf = (
        b\"%PDF-1.4\\n\"
        b\"1 0 obj\\n<< /Type /Catalog /Pages 2 0 R >>\\nendobj\\n\"
        b\"2 0 obj\\n<< /Type /Pages /Kids [] /Count 0 >>\\nendobj\\n\"
        b\"xref\\n0 3\\n0000000000 65535 f \\n0000000009 00000 n \\n0000000058 00000 n \\n\"
        b\"trailer\\n<< /Size 3 /Root 1 0 R >>\\nstartxref\\n115\\n%%EOF\\n\"
    )
    pdf_path.write_bytes(minimal_zero_page_pdf)
    return pdf_path


def test_missing_pdf_raises_not_found_error(tmp_path: Path) -> None:
    missing_path = tmp_path / \"does_not_exist.pdf\"
    with pytest.raises(PDFNotFoundError) as exc_info:
        load_pdf(missing_path)

    assert \"not found\" in str(exc_info.value).lower()
    assert isinstance(exc_info.value, FileNotFoundError)


def test_pages_extracted_and_page_numbers_preserved(
    multi_page_pdf_with_blanks: Path,
) -> None:
    pages = load_pdf(multi_page_pdf_with_blanks)
    assert len(pages) == 2

    page1 = pages[0]
    assert page1.page == 1
    assert \"Chapter 1\" in page1.text
    assert page1.source == \"sample_multipage.pdf\"
    assert page1.metadata == {\"source\": \"sample_multipage.pdf\", \"page\": 1}

    page3 = pages[1]
    assert page3.page == 3
    assert \"Chapter 2\" in page3.text
    assert page3.source == \"sample_multipage.pdf\"
    assert page3.metadata == {\"source\": \"sample_multipage.pdf\", \"page\": 3}


def test_empty_pages_ignored_and_empty_pdf_raises(whitespace_only_pdf: Path) -> None:
    with pytest.raises(EmptyPDFError) as exc_info:
        load_pdf(whitespace_only_pdf)
    assert \"contains no extractable text\" in str(exc_info.value)


def test_zero_page_pdf_raises_empty_error(zero_page_pdf: Path) -> None:
    with pytest.raises(EmptyPDFError) as exc_info:
        load_pdf(zero_page_pdf)
    assert \"0 pages\" in str(exc_info.value)


def test_corrupted_file_raises_pdf_read_error(tmp_path: Path) -> None:
    corrupt_file = tmp_path / \"corrupt.pdf\"
    corrupt_file.write_bytes(b\"%PDF-1.4 completely invalid corrupt binary stream\")

    with pytest.raises(PDFReadError) as exc_info:
        load_pdf(corrupt_file)

    assert \"Failed to open PDF file\" in str(exc_info.value)


def test_document_page_model_validation() -> None:
    doc_page = DocumentPage(text=\"Sample document text.\", page=1, source=\"Ebook-Agentic-AI.pdf\")
    assert doc_page.text == \"Sample document text.\"
    assert doc_page.page == 1
    assert doc_page.source == \"Ebook-Agentic-AI.pdf\"
    assert doc_page.metadata == {\"source\": \"Ebook-Agentic-AI.pdf\", \"page\": 1}


def test_actual_ebook_pdf_if_present() -> None:
    actual_pdf = Path(\"data/Ebook-Agentic-AI.pdf\")
    if not actual_pdf.exists():
        pytest.skip(\"data/Ebook-Agentic-AI.pdf is not present yet in environment.\")

    loader = PDFLoader(file_path=actual_pdf)
    pages = loader.load()

    assert len(pages) > 0
    for page in pages:
        assert isinstance(page.page, int)
        assert page.page >= 1
        assert page.source == \"Ebook-Agentic-AI.pdf\"
        assert len(page.text.strip()) > 0
        assert page.metadata[\"source\"] == \"Ebook-Agentic-AI.pdf\"
        assert page.metadata[\"page\"] == page.page
"""

FILES["tests/test_chunking.py"] = """\"\"\"Tests for document chunking pipeline.\"\"\"

import pytest

from app.config import get_settings
from app.ingestion.chunker import (
    DocumentChunk,
    DocumentChunker,
    chunk_documents,
)
from app.ingestion.loader import DocumentPage


@pytest.fixture
def sample_document_page() -> DocumentPage:
    content = (
        \"Agentic AI systems represent a significant evolution from passive LLM generation. \"
        \"Instead of producing text in a single feed-forward pass, an agentic system executes \"
        \"autonomous loops that perceive context, plan actions, and retrieve information.\\n\\n\"
        \"Retrieval-Augmented Generation (RAG) grounds language models in external knowledge bases. \"
        \"In a naive RAG system, documents are fetched and passed to the model unconditionally. \"
        \"However, in an agentic RAG architecture, retrieval is dynamic and verifiable.\\n\\n\"
        \"By evaluating retrieval relevance, detecting potential hallucinations, and rewriting \"
        \"queries when necessary, agentic RAG provides deterministic reliability for enterprise \"
        \"production workloads.\"
    )
    return DocumentPage(text=content, page=12, source=\"Ebook-Agentic-AI.pdf\")


def test_chunks_are_generated(sample_document_page: DocumentPage) -> None:
    chunks = chunk_documents([sample_document_page], chunk_size=200, chunk_overlap=30)
    assert len(chunks) > 1
    for chunk in chunks:
        assert isinstance(chunk, DocumentChunk)
        assert len(chunk.text) > 0


def test_chunk_size_is_respected_approximately(sample_document_page: DocumentPage) -> None:
    chunk_size = 250
    overlap = 30
    chunks = chunk_documents([sample_document_page], chunk_size=chunk_size, chunk_overlap=overlap)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.text) <= chunk_size + 20


def test_chunk_overlap_is_configured() -> None:
    continuous_text = (
        \"Sentence one explains agentic AI foundations. \"
        \"Sentence two discusses how autonomous workflows operate in production. \"
        \"Sentence three explores vector retrieval mechanics and dense indexing. \"
        \"Sentence four covers hallucination grading methods and grounding checks. \"
        \"Sentence five summarizes evaluation frameworks and benchmarks.\"
    )
    doc_page = DocumentPage(text=continuous_text, page=1, source=\"test.pdf\")
    chunk_size = 130
    chunk_overlap = 45

    chunks = chunk_documents([doc_page], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert len(chunks) >= 3
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i].text
        next_chunk = chunks[i + 1].text

        current_words = current_chunk.split()
        next_words = next_chunk.split()

        shared_words = set(current_words[-5:]) & set(next_words[:8])
        assert len(shared_words) > 0, f\"Expected overlapping words between chunk {i} and {i + 1}\"


def test_metadata_is_preserved(sample_document_page: DocumentPage) -> None:
    chunks = chunk_documents([sample_document_page], chunk_size=250, chunk_overlap=30)
    assert len(chunks) >= 2
    for i, chunk in enumerate(chunks, start=1):
        assert chunk.source == \"Ebook-Agentic-AI.pdf\"
        assert chunk.page == 12
        assert chunk.chunk_id == f\"page_12_chunk_{i:02d}\"
        assert chunk.metadata[\"source\"] == \"Ebook-Agentic-AI.pdf\"
        assert chunk.metadata[\"page\"] == 12
        assert chunk.metadata[\"chunk_id\"] == chunk.chunk_id


def test_chunk_ids_are_unique() -> None:
    pages = [
        DocumentPage(text=\"First page content.\" * 15, page=1, source=\"Ebook-Agentic-AI.pdf\"),
        DocumentPage(text=\"Second page content.\" * 15, page=2, source=\"Ebook-Agentic-AI.pdf\"),
        DocumentPage(text=\"Third page content.\" * 15, page=3, source=\"Ebook-Agentic-AI.pdf\"),
    ]
    chunks = chunk_documents(pages, chunk_size=200, chunk_overlap=30)
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    assert len(chunk_ids) > 5
    assert len(chunk_ids) == len(set(chunk_ids)), \"Duplicate chunk IDs found\"


def test_empty_text_is_ignored() -> None:
    empty_pages = [
        DocumentPage(text=\"   \\n\\t   \", page=1, source=\"Ebook-Agentic-AI.pdf\"),
        DocumentPage(text=\"\\n\\n\", page=2, source=\"Ebook-Agentic-AI.pdf\"),
    ]
    chunks = chunk_documents(empty_pages)
    assert chunks == []


def test_chunker_uses_settings_defaults() -> None:
    settings = get_settings()
    chunker = DocumentChunker()
    assert chunker.chunk_size == settings.CHUNK_SIZE
    assert chunker.chunk_overlap == settings.CHUNK_OVERLAP


def test_invalid_overlap_raises_value_error() -> None:
    with pytest.raises(ValueError) as exc_info:
        DocumentChunker(chunk_size=100, chunk_overlap=100)
    assert \"strictly less than chunk_size\" in str(exc_info.value)
"""

FILES["tests/test_embedder.py"] = """\"\"\"Unit tests for OpenAI embedding layer with mocked provider.\"\"\"

import logging
from unittest.mock import MagicMock
import pytest

from app.config import get_settings
from app.ingestion.chunker import DocumentChunk
from app.ingestion.embedder import (
    EmbeddedChunk,
    EmbeddingAPIError,
    EmbeddingService,
    MissingAPIKeyError,
    embed_chunks,
)


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id=\"page_12_chunk_01\",
            source=\"Ebook-Agentic-AI.pdf\",
            page=12,
            text=\"Autonomous loops perceive context, plan actions, and retrieve information.\",
            metadata={\"source\": \"Ebook-Agentic-AI.pdf\", \"page\": 12, \"chapter\": 1},
        ),
        DocumentChunk(
            chunk_id=\"page_12_chunk_02\",
            source=\"Ebook-Agentic-AI.pdf\",
            page=12,
            text=\"Agentic RAG evaluates retrieval relevance and mitigates hallucinations.\",
            metadata={\"source\": \"Ebook-Agentic-AI.pdf\", \"page\": 12, \"chapter\": 1},
        ),
    ]


@pytest.fixture
def mock_embeddings_client() -> MagicMock:
    mock_client = MagicMock()
    mock_client.embed_documents.return_value = [
        [0.012, -0.045, 0.089] * 512,
        [-0.034, 0.071, -0.015] * 512,
    ]
    mock_client.embed_query.return_value = [0.055, -0.021, 0.043] * 512
    return mock_client


def test_embedding_service_initializes_from_config(mock_embeddings_client: MagicMock) -> None:
    settings = get_settings()
    service = EmbeddingService(embeddings_client=mock_embeddings_client)
    assert service.model == settings.OPENAI_EMBEDDING_MODEL
    assert service.client == mock_embeddings_client


def test_missing_api_key_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(\"OPENAI_API_KEY\", \"\")
    get_settings.cache_clear()

    with pytest.raises(MissingAPIKeyError) as exc_info:
        EmbeddingService(api_key=\"\")

    assert \"OpenAI API key is missing\" in str(exc_info.value)
    get_settings.cache_clear()


def test_chunks_passed_to_embedding_layer(
    sample_chunks: list[DocumentChunk],
    mock_embeddings_client: MagicMock,
) -> None:
    service = EmbeddingService(embeddings_client=mock_embeddings_client)
    service.embed_chunks(sample_chunks)
    expected_texts = [chunk.text for chunk in sample_chunks]
    mock_embeddings_client.embed_documents.assert_called_once_with(expected_texts)


def test_returned_vectors_structure(
    sample_chunks: list[DocumentChunk],
    mock_embeddings_client: MagicMock,
) -> None:
    service = EmbeddingService(embeddings_client=mock_embeddings_client)
    embedded_chunks = service.embed_chunks(sample_chunks)
    assert len(embedded_chunks) == len(sample_chunks)
    for embedded in embedded_chunks:
        assert isinstance(embedded, EmbeddedChunk)
        assert isinstance(embedded.embedding, list)
        assert len(embedded.embedding) == 1536
        assert all(isinstance(v, float) for v in embedded.embedding)


def test_metadata_remains_associated(
    sample_chunks: list[DocumentChunk],
    mock_embeddings_client: MagicMock,
) -> None:
    service = EmbeddingService(embeddings_client=mock_embeddings_client)
    embedded_chunks = service.embed_chunks(sample_chunks)

    for original, embedded in zip(sample_chunks, embedded_chunks):
        assert embedded.chunk_id == original.chunk_id
        assert embedded.source == original.source
        assert embedded.page == original.page
        assert embedded.text == original.text
        assert embedded.metadata[\"chunk_id\"] == original.chunk_id
        assert embedded.metadata[\"source\"] == original.source
        assert embedded.metadata[\"page\"] == original.page
        assert embedded.metadata.get(\"chapter\") == 1


def test_api_errors_handled_cleanly(
    sample_chunks: list[DocumentChunk],
    mock_embeddings_client: MagicMock,
) -> None:
    mock_embeddings_client.embed_documents.side_effect = RuntimeError(\"OpenAI 429 Too Many Requests\")
    service = EmbeddingService(embeddings_client=mock_embeddings_client)

    with pytest.raises(EmbeddingAPIError) as exc_info:
        service.embed_chunks(sample_chunks)

    assert \"OpenAI embedding generation failed\" in str(exc_info.value)
    assert \"OpenAI 429\" in str(exc_info.value)


def test_embed_query_success_and_error(mock_embeddings_client: MagicMock) -> None:
    service = EmbeddingService(embeddings_client=mock_embeddings_client)
    vector = service.embed_query(\"What is Agentic RAG?\")
    assert len(vector) == 1536
    mock_embeddings_client.embed_query.assert_called_once_with(\"What is Agentic RAG?\")

    with pytest.raises(ValueError):
        service.embed_query(\"   \")


def test_empty_chunks_returns_empty_list(mock_embeddings_client: MagicMock) -> None:
    service = EmbeddingService(embeddings_client=mock_embeddings_client)
    result = service.embed_chunks([])
    assert result == []
    mock_embeddings_client.embed_documents.assert_not_called()


def test_never_log_api_key(
    caplog: pytest.LogCaptureFixture,
    mock_embeddings_client: MagicMock,
) -> None:
    secret_key = \"sk-super-confidential-secret-key-12345\"
    with caplog.at_level(logging.DEBUG):
        EmbeddingService(
            api_key=secret_key,
            embeddings_client=mock_embeddings_client,
        )

    for record in caplog.records:
        assert secret_key not in record.message
"""


def sync() -> None:
    for rel_path, content in FILES.items():
        target = BASE_DIR / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        print(f"Synced: {rel_path} ({len(content)} bytes)")


if __name__ == "__main__":
    sync()
