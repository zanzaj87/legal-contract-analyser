"""
Risk Assessor Agent.
Evaluates extracted clauses for legal risk and identifies missing protections.
"""

from models.state import ContractAnalysisState
from models.schemas import RiskAssessmentResult, ClauseExtractionResult
from utils.llm import get_llm
from agents.prompts import RISK_ASSESSOR_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage


def _format_clauses_for_assessment(extraction: ClauseExtractionResult) -> str:
    """Format extracted clauses into a readable string for the Risk Assessor."""
    parts = [
        f"Contract Type: {extraction.contract_type}",
        f"Parties: {', '.join(extraction.parties)}",
    ]
    if extraction.effective_date:
        parts.append(f"Effective Date: {extraction.effective_date}")

    parts.append(f"\nExtracted Clauses ({len(extraction.clauses)} found):")
    parts.append("-" * 60)

    for i, clause in enumerate(extraction.clauses, 1):
        parts.append(
            f"\n[{i}] {clause.clause_type.upper()} — {clause.title}"
            f"\n    Section: {clause.section_reference}"
            f"\n    Text: {clause.text}"
        )

    return "\n".join(parts)


def risk_assessor_agent(state: ContractAnalysisState) -> dict:
    """
    Node: Assess risk for each extracted clause.

    Reads: extraction_result
    Writes: risk_result, current_step
    """
    extraction = state.get("extraction_result")

    if not extraction:
        return {
            "current_step": "error",
            "error_message": "No extraction results available for risk assessment.",
        }

    try:
        llm = get_llm(model="gpt-5.2", temperature=0.0)
        structured_llm = llm.with_structured_output(RiskAssessmentResult)

        clauses_text = _format_clauses_for_assessment(extraction)

        result: RiskAssessmentResult = structured_llm.invoke(
            [
                SystemMessage(content=RISK_ASSESSOR_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"Assess the risk of the following contract clauses:\n\n"
                    f"{clauses_text}"
                ),
            ]
        )

        return {
            "risk_result": result,
            "current_step": "summarise",
        }

    except Exception as e:
        return {
            "current_step": "error",
            "error_message": f"Risk assessment failed: {str(e)}",
        }
