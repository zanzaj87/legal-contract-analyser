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
