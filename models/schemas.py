"""
Pydantic models for structured data throughout the pipeline.
These enforce schema consistency between agents.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ---------- Clause Extraction Models ----------

class ExtractedClause(BaseModel):
    """A single clause extracted from a contract."""

    clause_type: str = Field(
        description="Type of clause, e.g. 'indemnification', 'termination', "
        "'limitation_of_liability', 'confidentiality', 'governing_law', "
        "'force_majeure', 'representations_and_warranties', 'assignment', 'other'"
    )
    title: str = Field(description="Short title or heading of the clause")
    text: str = Field(description="Full text of the extracted clause")
    section_reference: str = Field(
        description="Section number or reference, e.g. 'Section 8.2' or 'Article IV'"
    )


class ClauseExtractionResult(BaseModel):
    """Structured output from the Clause Extractor agent."""

    clauses: List[ExtractedClause] = Field(default_factory=list)
    contract_type: str = Field(
        description="Type of contract, e.g. 'NDA', 'SaaS Agreement', 'Employment', 'MSA'"
    )
    parties: List[str] = Field(
        default_factory=list,
        description="Names of the parties to the contract",
    )
    effective_date: Optional[str] = Field(
        default=None, description="Effective date of the contract if found"
    )


# ---------- Risk Assessment Models ----------

class ClauseRiskAssessment(BaseModel):
    """Risk assessment for a single clause."""

    clause_type: str
    section_reference: str
    risk_level: Literal["low", "medium", "high"] = Field(
        description="Overall risk level for this clause"
    )
    risk_reasoning: str = Field(
        description="Explanation of why this risk level was assigned"
    )
    benchmark_comparison: str = Field(
        default="",
        description="How this clause compares to benchmark clauses from SEC EDGAR filings"
    )
    key_concerns: List[str] = Field(
        default_factory=list,
        description="Specific concerns or red flags identified",
    )
    recommendation: str = Field(
        description="Suggested action or negotiation point"
    )


class RiskAssessmentResult(BaseModel):
    """Structured output from the Risk Assessor agent."""

    overall_risk: Literal["low", "medium", "high"] = Field(
        description="Overall contract risk level"
    )
    clause_assessments: List[ClauseRiskAssessment] = Field(default_factory=list)
    missing_clauses: List[str] = Field(
        default_factory=list,
        description="Important clause types that are absent from the contract",
    )
    summary_of_concerns: str = Field(
        description="Brief narrative of the main risk themes"
    )


# ---------- Missing Clause Analysis Models ----------

class MissingClause(BaseModel):
    """A single clause that is expected but missing from the contract."""

    clause_type: str = Field(
        description="Type of the missing clause, e.g. 'Force Majeure', 'Data Protection'"
    )
    importance: Literal["critical", "recommended", "optional"] = Field(
        description="How important this clause is for this contract type"
    )
    risk_if_absent: str = Field(
        description="What risks arise from this clause being missing"
    )
    typical_coverage: str = Field(
        description="What this clause typically covers in similar contracts"
    )
    recommendation: str = Field(
        description="Specific action to address the gap"
    )


class MissingClauseResult(BaseModel):
    """Structured output from the Missing Clause Checker agent."""

    contract_type: str = Field(
        description="The type of contract being analysed"
    )
    clauses_found: List[str] = Field(
        default_factory=list,
        description="List of clause types that were found in the contract"
    )
    missing_clauses: List[MissingClause] = Field(
        default_factory=list,
        description="Clauses expected for this contract type but not found"
    )
    completeness_score: Literal["low", "medium", "high"] = Field(
        description="Overall completeness: low = many gaps, high = comprehensive"
    )
    summary: str = Field(
        description="Brief narrative of the main gaps and their implications"
    )


# ---------- Reviewer Models ----------

class ReviewResult(BaseModel):
    """Structured output from the Reviewer agent."""

    decision: Literal["approve", "revise_summary", "revise_risk"] = Field(
        description=(
            "approve: analysis is consistent and complete, proceed to final output. "
            "revise_summary: the summary has issues (missed key findings, misranked risks, "
            "inconsistent with underlying data) — send back to summariser. "
            "revise_risk: a risk assessment is incorrect or inconsistent — send back to "
            "risk assessor for re-evaluation, then re-summarise."
        )
    )
    quality_score: Literal["low", "medium", "high"] = Field(
        description="Overall quality of the analysis: low = major issues, high = ready to deliver"
    )
    issues_found: List[str] = Field(
        default_factory=list,
        description="Specific issues identified across the analysis outputs"
    )
    revision_instructions: str = Field(
        default="",
        description=(
            "If decision is revise_summary or revise_risk, provide specific instructions "
            "for what needs to change. Be precise — reference specific clauses, risk levels, "
            "or summary sections that need correction."
        )
    )