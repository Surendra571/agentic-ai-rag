"""Tests for PDF ingestion layer."""

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
    pdf_path = tmp_path / "sample_multipage.pdf"
    doc = pymupdf.open()

    p1 = doc.new_page()
    p1.insert_text((72, 72), "Chapter 1: Foundations of Agentic AI Systems.")
    doc.new_page()
    p3 = doc.new_page()
    p3.insert_text((72, 72), "Chapter 2: Retrieval-Augmented Generation Workflows.")

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def whitespace_only_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "whitespace_only.pdf"
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "   \n\t  \n  ")
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def zero_page_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "zero_pages.pdf"
    minimal_zero_page_pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
        b"xref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n"
        b"trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n115\n%%EOF\n"
    )
    pdf_path.write_bytes(minimal_zero_page_pdf)
    return pdf_path


def test_missing_pdf_raises_not_found_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.pdf"
    with pytest.raises(PDFNotFoundError) as exc_info:
        load_pdf(missing_path)

    assert "not found" in str(exc_info.value).lower()
    assert isinstance(exc_info.value, FileNotFoundError)


def test_pages_extracted_and_page_numbers_preserved(
    multi_page_pdf_with_blanks: Path,
) -> None:
    pages = load_pdf(multi_page_pdf_with_blanks)
    assert len(pages) == 2

    page1 = pages[0]
    assert page1.page == 1
    assert "Chapter 1" in page1.text
    assert page1.source == "sample_multipage.pdf"
    assert page1.metadata == {"source": "sample_multipage.pdf", "page": 1}

    page3 = pages[1]
    assert page3.page == 3
    assert "Chapter 2" in page3.text
    assert page3.source == "sample_multipage.pdf"
    assert page3.metadata == {"source": "sample_multipage.pdf", "page": 3}


def test_empty_pages_ignored_and_empty_pdf_raises(whitespace_only_pdf: Path) -> None:
    with pytest.raises(EmptyPDFError) as exc_info:
        load_pdf(whitespace_only_pdf)
    assert "contains no extractable text" in str(exc_info.value)


def test_zero_page_pdf_raises_empty_error(zero_page_pdf: Path) -> None:
    with pytest.raises(EmptyPDFError) as exc_info:
        load_pdf(zero_page_pdf)
    assert "0 pages" in str(exc_info.value)


def test_corrupted_file_raises_pdf_read_error(tmp_path: Path) -> None:
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_bytes(b"%PDF-1.4 completely invalid corrupt binary stream")

    with pytest.raises(PDFReadError) as exc_info:
        load_pdf(corrupt_file)

    assert "Failed to open PDF file" in str(exc_info.value)


def test_document_page_model_validation() -> None:
    doc_page = DocumentPage(text="Sample document text.", page=1, source="Ebook-Agentic-AI.pdf")
    assert doc_page.text == "Sample document text."
    assert doc_page.page == 1
    assert doc_page.source == "Ebook-Agentic-AI.pdf"
    assert doc_page.metadata == {"source": "Ebook-Agentic-AI.pdf", "page": 1}


def test_actual_ebook_pdf_if_present() -> None:
    actual_pdf = Path("data/Ebook-Agentic-AI.pdf")
    if not actual_pdf.exists():
        pytest.skip("data/Ebook-Agentic-AI.pdf is not present yet in environment.")

    loader = PDFLoader(file_path=actual_pdf)
    pages = loader.load()

    assert len(pages) > 0
    for page in pages:
        assert isinstance(page.page, int)
        assert page.page >= 1
        assert page.source == "Ebook-Agentic-AI.pdf"
        assert len(page.text.strip()) > 0
        assert page.metadata["source"] == "Ebook-Agentic-AI.pdf"
        assert page.metadata["page"] == page.page
