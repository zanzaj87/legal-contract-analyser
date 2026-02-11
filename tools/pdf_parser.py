"""
PDF parsing tool using PyMuPDF (fitz).
Extracts text and basic structure from contract PDFs.
"""

import fitz  # PyMuPDF
from pathlib import Path


def parse_pdf(pdf_path: str) -> dict:
    """
    Extract text and metadata from a PDF file.

    Returns:
        dict with keys:
            - full_text: str — complete extracted text
            - pages: list[dict] — per-page text and metadata
            - metadata: dict — file-level metadata
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)

    pages = []
    full_text_parts = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        pages.append(
            {
                "page_number": page_num + 1,
                "text": text,
                "char_count": len(text),
            }
        )
        full_text_parts.append(f"--- Page {page_num + 1} ---\n{text}")

    full_text = "\n\n".join(full_text_parts)

    metadata = {
        "filename": path.name,
        "page_count": len(doc),
        "total_chars": len(full_text),
        "pdf_metadata": dict(doc.metadata) if doc.metadata else {},
    }

    doc.close()

    return {
        "full_text": full_text,
        "pages": pages,
        "metadata": metadata,
    }
