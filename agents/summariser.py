"""
Summariser Agent.
Produces a clear executive summary combining extraction and risk results.
"""

from models.state import ContractAnalysisState
from models.schemas import ClauseExtractionResult, RiskAssessmentResult
from utils.llm import get_llm
from agents.prompts import SUMMARISER_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage


def _build_analysis_context(
    extraction: ClauseExtractionResult,
    risk: RiskAssessmentResult,
) -> str:
    """Combine extraction and risk data into a context string for the Summariser."""
    parts = [
        "=== CONTRACT DETAILS ===",
        f"Type: {extraction.contract_type}",
        f"Parties: {', '.join(extraction.parties)}",
        f"Effective Date: {extraction.effective_date or 'Not specified'}",
        f"Clauses Extracted: {len(extraction.clauses)}",
        "",
        "=== RISK OVERVIEW ===",
        f"Overall Risk: {risk.overall_risk.upper()}",
        f"Concerns: {risk.summary_of_concerns}",
        "",
    ]

    if risk.missing_clauses:
        parts.append(f"Missing Clauses: {', '.join(risk.missing_clauses)}")
        parts.append("")

    parts.append("=== CLAUSE-BY-CLAUSE ASSESSMENT ===")
    for assessment in risk.clause_assessments:
        parts.append(
            f"\n[{assessment.risk_level.upper()}] {assessment.clause_type} "
            f"({assessment.section_reference})"
            f"\n  Reasoning: {assessment.risk_reasoning}"
            f"\n  Concerns: {', '.join(assessment.key_concerns) if assessment.key_concerns else 'None'}"
            f"\n  Recommendation: {assessment.recommendation}"
        )

    return "\n".join(parts)


def summariser_agent(state: ContractAnalysisState) -> dict:
    """
    Node: Produce an executive summary of the full analysis.

    Reads: extraction_result, risk_result
    Writes: executive_summary, current_step
    """
    extraction = state.get("extraction_result")
    risk = state.get("risk_result")

    if not extraction or not risk:
        return {
            "current_step": "error",
            "error_message": "Missing extraction or risk data for summarisation.",
        }

    try:
        llm = get_llm(model="gpt-5.2", temperature=0.1)  # slight creativity for readable summary

        context = _build_analysis_context(extraction, risk)

        response = llm.invoke(
            [
                SystemMessage(content=SUMMARISER_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"Produce an executive summary from this analysis:\n\n"
                    f"{context}"
                ),
            ]
        )

        return {
            "executive_summary": response.content,
            "current_step": "complete",
        }

    except Exception as e:
        return {
            "current_step": "error",
            "error_message": f"Summarisation failed: {str(e)}",
        }
