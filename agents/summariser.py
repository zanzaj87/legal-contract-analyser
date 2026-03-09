"""
Summariser Agent.
Produces a clear executive summary combining extraction, risk, and missing clause results.
Receives inputs from TWO parallel agents: risk_assessor and missing_clause_checker.
"""

from models.state import ContractAnalysisState
from models.schemas import ClauseExtractionResult, RiskAssessmentResult, MissingClauseResult
from utils.llm import get_llm
from agents.prompts import SUMMARISER_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage


def _build_analysis_context(
    extraction: ClauseExtractionResult,
    risk: RiskAssessmentResult,
    missing: MissingClauseResult | None = None,
) -> str:
    """Combine extraction, risk, and missing clause data into a context string."""
    parts = [
        "=== CONTRACT DETAILS ===",
        f"Type: {extraction.contract_type}",
        f"Parties: {', '.join(extraction.parties)}",
        f"Effective Date: {extraction.effective_date or 'Not specified'}",
        f"Clauses Extracted: {len(extraction.clauses)}",
        "",
        "=== RISK ASSESSMENT (from Risk Assessor agent) ===",
        f"Overall Risk: {risk.overall_risk.upper()}",
        f"Concerns: {risk.summary_of_concerns}",
        "",
    ]

    parts.append("--- Clause-by-Clause Assessment ---")
    for assessment in risk.clause_assessments:
        parts.append(
            f"\n[{assessment.risk_level.upper()}] {assessment.clause_type} "
            f"({assessment.section_reference})"
            f"\n  Reasoning: {assessment.risk_reasoning}"
            f"\n  Concerns: {', '.join(assessment.key_concerns) if assessment.key_concerns else 'None'}"
            f"\n  Recommendation: {assessment.recommendation}"
        )

    # Add missing clause analysis if available
    if missing:
        parts.append("")
        parts.append("=== MISSING CLAUSE ANALYSIS (from Missing Clause Checker agent) ===")
        parts.append(f"Completeness Score: {missing.completeness_score.upper()}")
        parts.append(f"Summary: {missing.summary}")
        parts.append(f"Clauses Found: {', '.join(missing.clauses_found)}")
        parts.append("")

        if missing.missing_clauses:
            parts.append(f"--- Missing Clauses ({len(missing.missing_clauses)} identified) ---")
            for mc in missing.missing_clauses:
                parts.append(
                    f"\n[{mc.importance.upper()}] {mc.clause_type}"
                    f"\n  Risk If Absent: {mc.risk_if_absent}"
                    f"\n  Typical Coverage: {mc.typical_coverage}"
                    f"\n  Recommendation: {mc.recommendation}"
                )
    elif risk.missing_clauses:
        # Fallback: use risk assessor's missing clauses if checker didn't run
        parts.append("")
        parts.append(f"Missing Clauses (from risk assessor): {', '.join(risk.missing_clauses)}")

    return "\n".join(parts)


def summariser_agent(state: ContractAnalysisState) -> dict:
    """
    Node: Produce an executive summary of the full analysis.

    Receives outputs from TWO parallel agents:
    - risk_assessor → risk_result
    - missing_clause_checker → missing_clause_result

    Merges both into a single coherent executive summary.

    Reads: extraction_result, risk_result, missing_clause_result
    Writes: executive_summary, current_step
    """
    extraction = state.get("extraction_result")
    risk = state.get("risk_result")
    missing = state.get("missing_clause_result")

    if not extraction or not risk:
        return {
            "current_step": "error",
            "error_message": "Missing extraction or risk data for summarisation.",
        }

    try:
        llm = get_llm(model="gpt-5.2", temperature=0.1)

        context = _build_analysis_context(extraction, risk, missing)

        review = state.get("review_result")
        if review and review.revision_instructions:
            human_message = (
                f"Produce an executive summary from this analysis:\n\n"
                f"{context}"
                f"\n\n=== REVISION REQUESTED BY REVIEWER ===\n"
                f"The following issues were identified in the previous summary. "
                f"Address them in this revision:\n\n"
                f"{review.revision_instructions}"
            )
        else:
            human_message = (
                f"Produce an executive summary from this analysis:\n\n"
                f"{context}"
            )

        response = llm.invoke(
            [
                SystemMessage(content=SUMMARISER_SYSTEM_PROMPT),
                HumanMessage(
                    content = human_message
                    # content=f"Produce an executive summary from this analysis:\n\n"
                    # f"{context}"
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
