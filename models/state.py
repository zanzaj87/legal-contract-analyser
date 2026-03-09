"""
Graph state definition.
This TypedDict is the shared state that flows between all agents in the graph.
Each agent reads what it needs and writes its outputs back to this state.
"""

from typing import TypedDict, Optional, Literal, Annotated
from models.schemas import ClauseExtractionResult, RiskAssessmentResult, MissingClauseResult, ReviewResult
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def _reduce_current_step(existing: str | None, new: str | None) -> str:
    """Reducer for current_step: error takes priority over other values."""
    if new == "error" or existing == "error":
        return "error"
    return new or existing or "parse"


def _reduce_error_message(existing: str | None, new: str | None) -> str | None:
    """Reducer for error_message: keep the first error encountered."""
    if existing:
        return existing
    return new


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

    # -- Risk Assessor outputs (parallel track 1) --
    risk_result: Optional[RiskAssessmentResult]

    # -- Missing Clause Checker outputs (parallel track 2) --
    missing_clause_result: Optional[MissingClauseResult]

    # -- Summariser outputs --
    executive_summary: Optional[str]

    # -- Reviewer outputs --
    review_result: Optional[ReviewResult]
    revision_count: int  # tracks how many revisions to enforce max limit

    # -- Workflow control --
    # Valid values: "parse", "extract", "assess_risk", "check_missing",
    # "summarise", "review", "complete", "error"
    current_step: Annotated[str, _reduce_current_step]
    error_message: Annotated[Optional[str], _reduce_error_message]