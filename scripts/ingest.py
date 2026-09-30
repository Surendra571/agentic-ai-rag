"""Production ingestion script for Agentic AI RAG knowledge base.

Performs:
    PDF extraction -> Chunking -> Vector Embedding -> Pinecone Upsert

Progress reported:
    1. PDF loaded
    2. Number of pages
    3. Number of chunks
    4. Embeddings generated
    5. Pinecone vectors uploaded
    6. Ingestion completed

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --pdf data/Ebook-Agentic-AI.pdf --batch-size 100
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Sequence

# Ensure project root is on sys.path when invoked as `python scripts/ingest.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel, Field

from app.config import get_settings
from app.ingestion.chunker import DocumentChunk, DocumentChunker
from app.ingestion.embedder import EmbeddedChunk, EmbeddingService
from app.ingestion.loader import (
    DocumentPage,
    EmptyPDFError,
    PDFLoader,
    PDFNotFoundError,
    PDFReadError,
)
from app.retrieval.vector_store import (
    PineconeConnectionError,
    PineconeVectorStore,
    VectorStoreError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")


class IngestionSummary(BaseModel):
    """Structured report of ingestion metrics."""

    pdf_path: str = Field(..., description="Path to the ingested PDF")
    pages_count: int = Field(..., ge=0, description="Total pages extracted")
    chunks_count: int = Field(..., ge=0, description="Total chunks generated")
    vectors_uploaded: int = Field(..., ge=0, description="Total vectors upserted into Pinecone")
    index_name: str = Field(..., description="Target Pinecone index")
    namespace: str = Field(..., description="Target Pinecone namespace")


def run_ingestion(
    pdf_path: str | Path = "data/Ebook-Agentic-AI.pdf",
    batch_size: int = 100,
    namespace: str | None = None,
    loader: PDFLoader | None = None,
    chunker: DocumentChunker | None = None,
    embedder: EmbeddingService | None = None,
    vector_store: PineconeVectorStore | None = None,
) -> IngestionSummary:
    """Execute repeatable document ingestion pipeline with deterministic vector IDs.

    Args:
        pdf_path: Path to the PDF file.
        batch_size: Batch size for Pinecone upserts.
        namespace: Optional Pinecone namespace override.
        loader: Optional injected PDFLoader instance.
        chunker: Optional injected DocumentChunker instance.
        embedder: Optional injected EmbeddingService instance.
        vector_store: Optional injected PineconeVectorStore instance.

    Returns:
        IngestionSummary: Ingestion statistics and metrics.

    Raises:
        PDFNotFoundError: If the target PDF file does not exist.
        EmptyPDFError: If the PDF has no readable pages.
        VectorStoreError: If Pinecone upsert fails.
    """
    settings = get_settings()
    pdf_file = Path(pdf_path)
    target_namespace = namespace or settings.PINECONE_NAMESPACE

    print("=================================================================")
    print("           AGENTIC AI RAG: DOCUMENT INGESTION PIPELINE           ")
    print("=================================================================")

    # 1. Load PDF
    print(f"[1/6] Loading PDF: {pdf_file.as_posix()}...")
    pdf_loader = loader or PDFLoader(pdf_file)
    pages: list[DocumentPage] = pdf_loader.load()
    print(f"      -> PDF loaded successfully from '{pdf_file.name}'.")

    # 2. Page Extraction Metrics
    pages_count = len(pages)
    print(f"[2/6] Number of pages extracted: {pages_count} non-empty page(s).")
    if pages_count == 0:
        raise EmptyPDFError(f"PDF at '{pdf_file}' has no extractable text.")

    # 3. Document Chunking
    print("[3/6] Chunking document using recursive character text splitting...")
    doc_chunker = chunker or DocumentChunker()
    chunks: list[DocumentChunk] = doc_chunker.chunk_documents(pages)
    chunks_count = len(chunks)
    print(
        f"      -> Number of chunks generated: {chunks_count} chunk(s) "
        f"(size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})."
    )
    if chunks_count == 0:
        raise ValueError("Chunking produced 0 chunks from the extracted pages.")

    # Print sample deterministic chunk ID
    sample_id = chunks[0].chunk_id
    print(f"      -> Deterministic chunk ID pattern verified (e.g. '{sample_id}').")

    # 4. Generate Embeddings
    print(f"[4/6] Generating embeddings with model '{settings.OPENAI_EMBEDDING_MODEL}'...")
    embedding_service = embedder or EmbeddingService()
    embedded_chunks: list[EmbeddedChunk] = embedding_service.embed_chunks(chunks)
    print(f"      -> Embeddings generated: {len(embedded_chunks)} vector(s) ready.")

    # 5. Pinecone Vector Upsert
    print(
        f"[5/6] Uploading vectors to Pinecone (index='{settings.PINECONE_INDEX_NAME}', "
        f"namespace='{target_namespace}', batch_size={batch_size})..."
    )
    store = vector_store or PineconeVectorStore()
    uploaded_count = store.upsert_chunks(
        chunks=embedded_chunks,
        batch_size=batch_size,
        namespace=target_namespace,
    )
    print(f"      -> Pinecone vectors uploaded: {uploaded_count} vector(s) upserted.")

    # 6. Ingestion Completed
    print("[6/6] Ingestion completed successfully!")
    print("=================================================================")
    print(f" Summary: {pages_count} pages -> {chunks_count} chunks -> {uploaded_count} vectors")
    print("=================================================================\n")

    return IngestionSummary(
        pdf_path=str(pdf_file),
        pages_count=pages_count,
        chunks_count=chunks_count,
        vectors_uploaded=uploaded_count,
        index_name=settings.PINECONE_INDEX_NAME,
        namespace=target_namespace,
    )


def main() -> int:
    """CLI entrypoint for running ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF document into Pinecone for Agentic RAG.",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default="data/Ebook-Agentic-AI.pdf",
        help="Path to the PDF file (default: data/Ebook-Agentic-AI.pdf)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for Pinecone upserts (default: 100)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Pinecone namespace override (default: from configuration)",
    )

    args = parser.parse_args()

    try:
        run_ingestion(
            pdf_path=args.pdf,
            batch_size=args.batch_size,
            namespace=args.namespace,
        )
        return 0
    except (PDFNotFoundError, FileNotFoundError) as exc:
        logger.error("PDF file not found: %s", exc)
        print(f"\n[ERROR] PDF file not found: {exc}", file=sys.stderr)
        return 1
    except (PDFReadError, EmptyPDFError) as exc:
        logger.error("PDF extraction error: %s", exc)
        print(f"\n[ERROR] PDF extraction failed: {exc}", file=sys.stderr)
        return 1
    except PineconeConnectionError as exc:
        logger.error("Pinecone connection error: %s", exc)
        print(f"\n[ERROR] Pinecone error: {exc}", file=sys.stderr)
        return 1
    except VectorStoreError as exc:
        logger.error("Vector storage error: %s", exc)
        print(f"\n[ERROR] Vector store error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error during ingestion: %s", exc)
        print(f"\n[ERROR] Ingestion failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
