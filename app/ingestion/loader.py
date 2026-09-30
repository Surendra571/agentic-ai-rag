"""PDF document loader and extractor module using PyMuPDF."""

from pathlib import Path
from typing import Any
import pymupdf
from pydantic import BaseModel, Field


class PDFIngestionError(Exception):
    """Base exception for all PDF ingestion errors."""


class PDFNotFoundError(PDFIngestionError, FileNotFoundError):
    """Raised when the specified PDF file cannot be found."""


class PDFReadError(PDFIngestionError):
    """Raised when the PDF file cannot be read, opened, or is corrupt."""


class EmptyPDFError(PDFIngestionError):
    """Raised when a PDF file has no pages or contains no extractable text."""


class DocumentPage(BaseModel):
    """Structured representation of an extracted document page."""

    text: str = Field(..., min_length=1, description="Extracted textual content of the page")
    page: int = Field(..., ge=1, description="1-based page number")
    source: str = Field(..., description="Source document filename or identifier")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")

    def model_post_init(self, __context: Any) -> None:
        """Populate default metadata if not explicitly provided."""
        if not self.metadata:
            self.metadata = {
                "source": self.source,
                "page": self.page,
            }


class PDFLoader:
    """Safely loads and extracts text page-by-page from PDF documents using PyMuPDF."""

    def __init__(self, file_path: str | Path = "data/Ebook-Agentic-AI.pdf") -> None:
        self.file_path = Path(file_path)

    def load(self) -> list[DocumentPage]:
        if not self.file_path.exists():
            raise PDFNotFoundError(f"PDF file not found at: {self.file_path}")

        if not self.file_path.is_file():
            raise PDFNotFoundError(f"Path is not a regular file: {self.file_path}")

        try:
            doc = pymupdf.open(self.file_path)
        except Exception as exc:
            raise PDFReadError(f"Failed to open PDF file '{self.file_path}': {exc}") from exc

        try:
            if len(doc) == 0:
                raise EmptyPDFError(f"PDF file '{self.file_path.name}' contains 0 pages.")

            source_name = self.file_path.name
            pages: list[DocumentPage] = []

            for page_index in range(len(doc)):
                page = doc[page_index]
                try:
                    text = page.get_text()
                except Exception as exc:
                    raise PDFReadError(
                        f"Failed to extract text from page {page_index + 1} of '{self.file_path.name}': {exc}"
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
                            "source": source_name,
                            "page": page_number,
                        },
                    )
                )

            if not pages:
                raise EmptyPDFError(
                    f"PDF file '{self.file_path.name}' has {len(doc)} pages, but contains no extractable text."
                )

            return pages
        finally:
            doc.close()


def load_pdf(file_path: str | Path = "data/Ebook-Agentic-AI.pdf") -> list[DocumentPage]:
    loader = PDFLoader(file_path=file_path)
    return loader.load()
