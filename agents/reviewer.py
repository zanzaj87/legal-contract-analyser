"""
Reviewer Agent.
Checks consistency and quality across all analysis outputs.
Can approve the final result or send it back for revision:
  - revise_summary: only the summariser reruns
  - revise_risk: the risk assessor reruns, then the summariser reruns

Limited to 1 revision to prevent infinite loops.
"""

from models.state import ContractAnalysisState
from models.schemas import (
    ReviewResult,
    ClauseExtractionResult,
    RiskAssessmentResult,
    MissingClauseResult,
)
from utils.llm import get_llm
from agents.prompts import REVIEWER_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage


MAX_REVISIONS = 1


def _build_review_context(state: ContractAnalysisState) -> str:
    """Build a comprehensive context string from all pipeline outputs for review."""
    extraction = state.get("extraction_result")
    risk = state.get("risk_result")
    missing = state.get("missing_clause_result")
    summary = state.get("executive_summary", "")

    parts = []

    # ─── Executive Summary ───
    parts.append("=== EXECUTIVE SUMMARY (produced by Summariser agent) ===")
    parts.append(summary if summary else "[No summary produced]")
    parts.append("")

    # ─── Extraction Overview ───
    if extraction:
        parts.append("=== EXTRACTION OVERVIEW ===")
        parts.append(f"Contract Type: {extraction.contract_type}")
        parts.append(f"Parties: {', '.join(extraction.parties)}")
        parts.append(f"Effective Date: {extraction.effective_date or 'Not specified'}")
        parts.append(f"Clauses Extracted: {len(extraction.clauses)}")
        parts.append(f"Clause Types: {', '.join(c.clause_type for c in extraction.clauses)}")
        parts.append("")

    # ─── Risk Assessment Summary ───
    if risk:
        parts.append("=== RISK ASSESSMENT (from Risk Assessor agent) ===")
        parts.append(f"Overall Risk: {risk.overall_risk.upper()}")
        parts.append(f"Summary: {risk.summary_of_concerns}")
        parts.append("")

        # Count risk levels
        high = sum(1 for a in risk.clause_assessments if a.risk_level == "high")
        medium = sum(1 for a in risk.clause_assessments if a.risk_level == "medium")
        low = sum(1 for a in risk.clause_assessments if a.risk_level == "low")
        parts.append(f"Risk Distribution: {high} HIGH, {medium} MEDIUM, {low} LOW")
        parts.append("")

        # List high-risk clauses for easy review
        if high > 0:
            parts.append("HIGH-RISK clauses:")
            for a in risk.clause_assessments:
                if a.risk_level == "high":
                    parts.append(
                        f"  - {a.clause_type} ({a.section_reference}): "
                        f"{a.risk_reasoning[:200]}..."
                    )
            parts.append("")

    # ─── Missing Clause Summary ───
    if missing:
        parts.append("=== MISSING CLAUSE ANALYSIS (from Missing Clause Checker agent) ===")
        parts.append(f"Completeness Score: {missing.completeness_score.upper()}")
        parts.append(f"Summary: {missing.summary}")
        critical = sum(1 for m in missing.missing_clauses if m.importance == "critical")
        recommended = sum(1 for m in missing.missing_clauses if m.importance == "recommended")
        optional = sum(1 for m in missing.missing_clauses if m.importance == "optional")
        parts.append(
            f"Missing: {critical} CRITICAL, {recommended} RECOMMENDED, {optional} OPTIONAL"
        )
        if critical > 0:
            parts.append("CRITICAL missing clauses:")
            for m in missing.missing_clauses:
                if m.importance == "critical":
                    parts.append(f"  - {m.clause_type}: {m.risk_if_absent[:150]}...")
        parts.append("")

    # ─── Revision Context ───
    revision_count = state.get("revision_count", 0)
    if revision_count > 0:
        parts.append(f"=== REVISION CONTEXT ===")
        parts.append(f"This is revision #{revision_count}.")
        prev_review = state.get("review_result")
        if prev_review:
            parts.append(f"Previous review decision: {prev_review.decision}")
            parts.append(f"Previous issues: {', '.join(prev_review.issues_found)}")
            parts.append(f"Previous instructions: {prev_review.revision_instructions}")
        parts.append("")

    return "\n".join(parts)


def reviewer_agent(state: ContractAnalysisState) -> dict:
    """
    Node: Review the complete analysis for consistency and quality.

    Checks:
    1. Does the executive summary accurately reflect the risk assessment?
    2. Are HIGH-risk items prominently featured in the summary?
    3. Are CRITICAL missing clauses highlighted?
    4. Is the overall risk rating consistent with the clause assessments?
    5. Are the recommended actions practical and prioritised correctly?

    Decisions:
    - approve: everything is consistent → proceed to END
    - revise_summary: summary has issues → send back to summariser
    - revise_risk: risk assessment has issues → send back to risk assessor

    After MAX_REVISIONS, always approves to prevent infinite loops.

    Reads: extraction_result, risk_result, missing_clause_result, executive_summary,
           review_result (previous), revision_count
    Writes: review_result, revision_count, current_step
    """
    revision_count = state.get("revision_count", 0)

    # Force approval after max revisions
    if revision_count >= MAX_REVISIONS:
        print(f"[REVIEWER] Max revisions ({MAX_REVISIONS}) reached — forcing approval.")
        return {
            "review_result": ReviewResult(
                decision="approve",
                quality_score="medium",
                issues_found=["Max revision limit reached — approving as-is."],
                revision_instructions="",
            ),
            "current_step": "complete",
        }

    try:
        llm = get_llm(model="gpt-5-mini", temperature=0.0)
        structured_llm = llm.with_structured_output(ReviewResult)

        context = _build_review_context(state)

        result: ReviewResult = structured_llm.invoke(
            [
                SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"Review the following contract analysis for consistency "
                    f"and quality:\n\n{context}"
                ),
            ]
        )

        print(f"[REVIEWER] Decision: {result.decision} | Quality: {result.quality_score}")
        if result.issues_found:
            for issue in result.issues_found:
                print(f"  - {issue}")

        # Determine next step based on decision
        if result.decision == "approve":
            next_step = "complete"
        elif result.decision == "revise_summary":
            next_step = "summarise"
        elif result.decision == "revise_risk":
            next_step = "assess_risk"
        else:
            next_step = "complete"  # Safety fallback

        return {
            "review_result": result,
            "revision_count": revision_count + 1,
            "current_step": next_step,
        }

    except Exception as e:
        print(f"[REVIEWER] Error: {e}")
        # If reviewer fails, approve anyway — don't block the pipeline
        return {
            "review_result": ReviewResult(
                decision="approve",
                quality_score="medium",
                issues_found=[f"Reviewer error: {str(e)}"],
                revision_instructions="",
            ),
            "current_step": "complete",
        }