"""
Parser Agent — Multi-Format Version.
Uses LLM tool calling to dynamically select the right extraction method
based on file type and content characteristics.
"""

from typing import Annotated
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from models.state import ContractAnalysisState
from utils.llm import get_llm
from agents.prompts import PARSER_SYSTEM_PROMPT

import fitz  # PyMuPDF
from pathlib import Path


# ---------- Define Tools ----------
# Each tool handles a different document format.
# The LLM will decide which one to call based on the file extension and context.


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


# ---------- Agent Node ----------

MULTI_FORMAT_PARSER_PROMPT = """\
You are a document parsing specialist. You have access to tools for extracting text \
from different file formats.

Given a file path, determine the best extraction method:
- Use `parse_pdf` for standard PDF files with selectable text
- Use `parse_docx` for Word documents (.docx, .doc)
- Use `ocr_scanned_document` for scanned PDFs (image-based) or image files (.png, .jpg)

Look at the file extension to decide. If a PDF extraction returns very little text \
relative to the page count, the PDF might be scanned — try OCR as a fallback.

After extraction, confirm whether the document appears to be a legal contract."""


def parser_agent(state: ContractAnalysisState) -> dict:
    """
    Node: Use tool calling to parse the uploaded document.

    The LLM decides which parsing tool to use based on the file type.
    This node returns an AIMessage with tool_calls — the ToolNode
    in the graph will execute the actual tool.
    """
    pdf_path = state["pdf_path"]
    file_ext = Path(pdf_path).suffix.lower()

    llm = get_llm(model="gpt-4o-mini", temperature=0.0)
    llm_with_tools = llm.bind_tools(PARSER_TOOLS)

    response = llm_with_tools.invoke(
        [
            SystemMessage(content=MULTI_FORMAT_PARSER_PROMPT),
            HumanMessage(
                content=f"Please extract text from this file: {pdf_path}\n"
                f"File extension: {file_ext}"
            ),
        ]
    )

    # The response will contain tool_calls if the LLM wants to use a tool.
    # The ToolNode in the graph will execute them and return results.
    # We store the messages so the LLM can see the tool results on the next pass.
    return {"messages": [response]}
