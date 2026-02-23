"""
Graph state definition.
This TypedDict is the shared state that flows between all agents in the graph.
Each agent reads what it needs and writes its outputs back to this state.
"""

from typing import TypedDict, Optional, Literal, Annotated
from models.schemas import ClauseExtractionResult, RiskAssessmentResult
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ContractAnalysisState(TypedDict):
    """Shared state for the contract analysis pipeline."""

    # -- Input --
    file_path: str

    # -- Messages (required for tool calling) --
    messages: Annotated[list[AnyMessage], add_messages]

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
