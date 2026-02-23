"""
Parser Agent.
Extracts text from PDF and validates it's a legal contract.
"""

from models.state import ContractAnalysisState
from tools.pdf_parser import parse_pdf
from utils.llm import get_llm
from agents.prompts import PARSER_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage


def parser_agent(state: ContractAnalysisState) -> dict:
    """
    Node: Parse the PDF and validate it's a contract.

    Reads: pdf_path
    Writes: parsed_text, document_metadata, current_step
    """
    pdf_path = state["file_path"]

    try:
        # Step 1: Extract text from PDF
        result = parse_pdf(pdf_path)
        full_text = result["full_text"]
        metadata = result["metadata"]

        if not full_text.strip():
            return {
                "parsed_text": None,
                "document_metadata": metadata,
                "current_step": "error",
                "error_message": "PDF appears to be empty or image-only (OCR not implemented).",
            }

        # Step 2: Quick validation with LLM — is this actually a contract?
        llm = get_llm(model="gpt-4o-mini", temperature=0.0)  # cheap model for validation
        response = llm.invoke(
            [
                SystemMessage(content=PARSER_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"Analyse this document text (first 3000 chars):\n\n"
                    f"{full_text[:3000]}"
                ),
            ]
        )

        metadata["parser_analysis"] = response.content

        return {
            "parsed_text": full_text,
            "document_metadata": metadata,
            "current_step": "extract",
        }

    except FileNotFoundError as e:
        return {
            "current_step": "error",
            "error_message": str(e),
        }
    except Exception as e:
        return {
            "current_step": "error",
            "error_message": f"Parser failed: {str(e)}",
        }
