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

IMPORTANT: Once you have successfully extracted text from the document, DO NOT call \
any more tools. Instead, respond with a brief assessment of whether the document \
appears to be a legal contract. Never call the same tool twice on the same file."""


def parser_agent(state: ContractAnalysisState) -> dict:
    """
    Node: Use tool calling to parse the uploaded document.

    The LLM decides which parsing tool to use based on the file type.
    This node returns an AIMessage with tool_calls — the ToolNode
    in the graph will execute the actual tool.
    """
    file_path = state["file_path"]
    file_ext = Path(file_path).suffix.lower()

    llm = get_llm(model="gpt-4o-mini", temperature=0.0)
    llm_with_tools = llm.bind_tools(PARSER_TOOLS)

    # Check if we already have messages (i.e. we're in a loop iteration)
    existing_messages = state.get("messages", [])

    if existing_messages:
        # We're looping back — the LLM needs to see its previous
        # tool calls and the tool results to decide what to do next
        response = llm_with_tools.invoke(existing_messages)
    else:
        # First invocation — send the initial prompt
        response = llm_with_tools.invoke(
            [
                SystemMessage(content=MULTI_FORMAT_PARSER_PROMPT),
                HumanMessage(
                    content=f"Please extract text from this file: {file_path}\n"
                    f"File extension: {file_ext}"
                ),
            ]
        )

    # If the LLM is NOT calling any more tools, it means parsing is done.
    # Extract the parsed text from the tool results in messages.
    if not response.tool_calls:
        # Find the tool result message (contains the extracted text)
        parsed_text = None
        for msg in existing_messages:
            if hasattr(msg, 'type') and msg.type == 'tool':
                parsed_text = msg.content

        return {
            "messages": [response],
            "parsed_text": parsed_text,
            "current_step": "extract" if parsed_text else "error",
            "error_message": None if parsed_text else "Failed to extract text from document.",
        }

    return {"messages": [response]}