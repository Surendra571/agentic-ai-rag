"""Document chunking module for vector retrieval using LangChain text splitting."""

from collections.abc import Sequence
from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from app.config import get_settings
from app.ingestion.loader import DocumentPage


class DocumentChunk(BaseModel):
    """Structured representation of a document chunk."""

    chunk_id: str = Field(..., description="Unique chunk identifier (e.g. page_12_chunk_03)")
    source: str = Field(..., description="Source document identifier or filename")
    page: int = Field(..., ge=1, description="Original 1-based page number")
    text: str = Field(..., min_length=1, description="Textual content of the chunk")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")

    def model_post_init(self, __context: Any) -> None:
        if not self.metadata:
            self.metadata = {
                "chunk_id": self.chunk_id,
                "source": self.source,
                "page": self.page,
            }


class DocumentChunker:
    """Chunks DocumentPage instances into structured DocumentChunk objects."""

    DEFAULT_SEPARATORS: list[str] = [
        "\n\n",
        "\n",
        " ",
        "",
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
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})"
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
            raw_text = str(page.get("text", ""))
            page_num = int(page.get("page", 1))
            source = str(page.get("source", "unknown"))
        else:
            raw_text = str(getattr(page, "text", ""))
            page_num = int(getattr(page, "page", 1))
            source = str(getattr(page, "source", "unknown"))

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

            chunk_id = f"page_{page_num:02d}_chunk_{chunk_idx:02d}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source=source,
                    page=page_num,
                    text=stripped,
                    metadata={
                        "chunk_id": chunk_id,
                        "source": source,
                        "page": page_num,
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
                    while f"{final_id}_{counter:02d}" in seen_ids:
                        counter += 1
                    final_id = f"{final_id}_{counter:02d}"
                    chunk.chunk_id = final_id
                    chunk.metadata["chunk_id"] = final_id

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
