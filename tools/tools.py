# ---------- Define Tools ----------
# Each tool handles a different document format.
# The LLM will decide which one to call based on the file extension and context.

from langchain_core.tools import tool
import fitz  # PyMuPDF
from pathlib import Path


@tool
def parse_pdf(file_path: str) -> dict:
    """
    Extract text from a PDF document. Use this for .pdf files
    that contain selectable text (not scanned images).
    """
    doc = fitz.open(file_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page_number": page_num + 1, "text": text})

    full_text = "\n\n".join(
        f"--- Page {p['page_number']} ---\n{p['text']}" for p in pages
    )
    doc.close()

    return {
        "full_text": full_text,
        "page_count": len(pages),
        "method": "pdf_text_extraction",
    }


@tool
def parse_docx(file_path: str) -> dict:
    """
    Extract text from a Word document (.docx).
    Use this for .docx or .doc files.
    """
    try:
        import docx

        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs)

        return {
            "full_text": full_text,
            "paragraph_count": len(paragraphs),
            "method": "docx_extraction",
        }
    except ImportError:
        return {"error": "python-docx not installed. Run: pip install python-docx"}


@tool
def ocr_scanned_document(file_path: str) -> dict:
    """
    Extract text from a scanned PDF or image file using OCR.
    Use this when the PDF contains images of text rather than selectable text,
    or for image files (.png, .jpg, .tiff).
    """
    try:
        import pytesseract
        from PIL import Image

        path = Path(file_path)

        # If PDF, convert pages to images first
        if path.suffix.lower() == ".pdf":
            doc = fitz.open(file_path)
            full_text_parts = []
            for page_num, page in enumerate(doc):
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
                full_text_parts.append(
                    f"--- Page {page_num + 1} ---\n{text}"
                )
            doc.close()
            full_text = "\n\n".join(full_text_parts)
        else:
            # Direct image file
            img = Image.open(file_path)
            full_text = pytesseract.image_to_string(img)

        return {
            "full_text": full_text,
            "method": "ocr",
        }
    except ImportError:
        return {
            "error": "OCR dependencies not installed. "
            "Run: pip install pytesseract pillow"
        }


# Collect all parser tools
PARSER_TOOLS = [parse_pdf, parse_docx, ocr_scanned_document]