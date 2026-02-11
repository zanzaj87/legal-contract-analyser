"""
Graph state definition.
This TypedDict is the shared state that flows between all agents in the graph.
Each agent reads what it needs and writes its outputs back to this state.
"""

from typing import TypedDict, Optional, Literal
from models.schemas import ClauseExtractionResult, RiskAssessmentResult


class ContractAnalysisState(TypedDict):
    """Shared state for the contract analysis pipeline."""

    # -- Input --
    pdf_path: str

    # -- Parser Agent outputs --
    parsed_text: Optional[str]
    document_metadata: Optional[dict]  # page count, file size, etc.

    # -- Clause Extractor outputs --
    extraction_result: Optional[ClauseExtractionResult]

    # -- Risk Assessor outputs --
    risk_result: Optional[RiskAssessmentResult]

    # -- Summariser outputs --
    executive_summary: Optional[str]

    # -- Workflow control --
    current_step: Literal[
        "parse", "extract", "assess_risk", "summarise", "complete", "error"
    ]
    error_message: Optional[str]
