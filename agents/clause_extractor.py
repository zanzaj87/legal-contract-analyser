"""
Clause Extractor Agent.
Identifies and extracts key legal clauses using structured LLM output.
"""

from models.state import ContractAnalysisState
from models.schemas import ClauseExtractionResult
from utils.llm import get_llm
from agents.prompts import CLAUSE_EXTRACTOR_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage


def clause_extractor_agent(state: ContractAnalysisState) -> dict:
    """
    Node: Extract structured clauses from the parsed contract text.

    Reads: parsed_text
    Writes: extraction_result, current_step
    """
    parsed_text = state.get("parsed_text", "")

    if not parsed_text:
        return {
            "current_step": "error",
            "error_message": "No parsed text available for clause extraction.",
        }

    try:
        llm = get_llm(model="gpt-5.2", temperature=0.0)

        # Use structured output — this forces the LLM to return a valid Pydantic object
        structured_llm = llm.with_structured_output(ClauseExtractionResult)

        # For long contracts, we might need to chunk — but for demo, send full text
        # In production, you'd implement a chunking strategy here
        # e.g. section-aware chunking, or process page-by-page and merge
        max_chars = 1_500_000  # ~375k tokens, safe for gpt-5.2's context
        text_to_analyse = parsed_text[:max_chars]

        if len(parsed_text) > max_chars:
            text_to_analyse += (
                f"\n\n[NOTE: Contract truncated at {max_chars} characters. "
                f"Full contract is {len(parsed_text)} characters.]"
            )

        result: ClauseExtractionResult = structured_llm.invoke(
            [
                SystemMessage(content=CLAUSE_EXTRACTOR_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"Extract all key clauses from this contract:\n\n"
                    f"{text_to_analyse}"
                ),
            ]
        )

        return {
            "extraction_result": result,
            "current_step": "assess_risk",
        }

    except Exception as e:
        return {
            "current_step": "error",
            "error_message": f"Clause extraction failed: {str(e)}",
        }
