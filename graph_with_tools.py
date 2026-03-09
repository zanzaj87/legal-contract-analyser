"""
LangGraph pipeline — Multi-Format Version with Tool Calling,
Parallel Agents, and Reviewer Feedback Loop.

Architecture:
  parser_agent → clause_extractor → fan-out → risk_assessor (+ RAG)        → fan-in
                                             → missing_clause_checker (+ RAG)
  → summariser → reviewer → approve → END
                           → revise_summary → summariser → reviewer (forced approve) → END
                           → revise_risk → risk_assessor → summariser → reviewer (forced approve) → END
"""

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from models.state import ContractAnalysisState
from agents.parser_with_tools import parser_agent, PARSER_TOOLS
from agents.clause_extractor import clause_extractor_agent
from agents.risk_assessor import risk_assessor_agent
from agents.missing_clause_checker import missing_clause_checker_agent
from agents.summariser import summariser_agent
from agents.reviewer import reviewer_agent


def parser_routing(state: ContractAnalysisState) -> str:
    """Route after parser agent: check for tool calls OR error."""
    # If the parser set an error (e.g. not a contract), go to error handler
    if state.get("current_step") == "error":
        return "error_handler"
    
    # Otherwise, check for tool calls (standard ReAct routing)
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return "clause_extractor"


def route_or_error(success_node: str):
    def router(state: ContractAnalysisState) -> str:
        if state.get("current_step") == "error":
            return "error_handler"
        return success_node
    return router


def fan_out_after_extraction(state: ContractAnalysisState) -> list[str]:
    """
    After clause extraction, fan out to both analysis agents in parallel.
    Returns a list of node names to execute simultaneously.
    """
    if state.get("current_step") == "error":
        return ["error_handler"]
    return ["risk_assessor", "missing_clause_checker"]


def reviewer_routing(state: ContractAnalysisState) -> str:
    """
    Route after reviewer: approve, revise summary, or revise risk assessment.

    The reviewer writes its decision to review_result.decision:
      - "approve" → END
      - "revise_summary" → summariser (rerun summary only)
      - "revise_risk" → risk_assessor (rerun risk, then summary)
    """
    if state.get("current_step") == "error":
        return "error_handler"

    review = state.get("review_result")
    if not review:
        return "complete"  # No review result = approve by default

    if review.decision == "revise_summary":
        print(f"[GRAPH] Reviewer requested summary revision")
        return "summariser"
    elif review.decision == "revise_risk":
        print(f"[GRAPH] Reviewer requested risk assessment revision")
        return "risk_assessor"
    else:
        return "complete"


def error_handler(state: ContractAnalysisState) -> dict:
    error_msg = state.get("error_message", "Unknown error occurred.")
    print(f"[ERROR] Pipeline failed: {error_msg}")
    return {"current_step": "error"}


def build_graph() -> StateGraph:
    """
    Construct the LangGraph pipeline with tool calling, parallel agents,
    and reviewer feedback loop.

    FULL ARCHITECTURE:

        ┌──────────────────────┐
        │  parser_agent        │◄─────────┐
        │  (LLM reasons,       │          │
        │   emits tool_calls)  │          │ loop back with
        └──────┬───────────────┘          │ tool results
               │                          │
               ▼                          │
        ┌──────────────────────┐          │
        │  parser_routing      │          │
        │  (3-way decision)    │          │
        └──┬───────┬───────┬───┘          │
           │       │       │              │
       not a    no tools  has tools       │
       contract    │       │              │
           │       ▼       ▼              │
           │  clause_  ┌───────────────┐  │
           │  extractor│  parser_tools │──┘
           │       │   │  (executes    │
           │       │   │   the tool)   │
           │       │   └───────────────┘
           │       │
           │       ▼ (fan-out: parallel execution)
           │       ┌─────────────────────────────────┐
           │       │                                 │
           │       ▼                                 ▼
           │  risk_assessor ◄── ChromaDB    missing_clause_checker ◄── ChromaDB
           │  (assess existing    (CUAD)    (identify gaps)              (CUAD)
           │   clauses)
           │       │                                 │
           │       └────────────┬────────────────────┘
           │                    ▼ (fan-in: both complete)
           │               summariser ◄──────────────────────┐
           │                    │                            │
           │                    ▼                            │
           │               reviewer                          │
           │                    │                            │
           │            ┌───────┼───────┐                    │
           │            │       │       │                    │
           │         approve  revise  revise                 │
           │            │     summary  risk                  │
           │            ▼       │       │                    │
           │          END       │       ▼                    │
           │                    │  risk_assessor             │
           │                    │       │                    │
           │                    └───────┴────────────────────┘
           │                    (max 1 revision, then forced approve)
           │
           ▼
        ┌──────────────────────┐
        │  error_handler       │──→ END
        │  "Not a contract"    │
        └──────────────────────┘
    """

    graph = StateGraph(ContractAnalysisState)

    # -- Nodes --
    graph.add_node("parser_agent", parser_agent)
    graph.add_node("parser_tools", ToolNode(tools=PARSER_TOOLS))
    graph.add_node("clause_extractor", clause_extractor_agent)
    graph.add_node("risk_assessor", risk_assessor_agent)
    graph.add_node("missing_clause_checker", missing_clause_checker_agent)
    graph.add_node("summariser", summariser_agent)
    graph.add_node("reviewer", reviewer_agent)
    graph.add_node("error_handler", error_handler)

    # -- Entry point --
    graph.set_entry_point("parser_agent")

    # -- Parser agent: ReAct loop with contract validation --
    graph.add_conditional_edges(
        "parser_agent",
        parser_routing,
        {
            "tools": "parser_tools",
            "clause_extractor": "clause_extractor",
            "error_handler": "error_handler",
        },
    )

    # After tools execute, loop back to the parser agent
    # so the LLM can see the results and decide what to do next
    graph.add_edge("parser_tools", "parser_agent")


    # -- Fan-out: clause_extractor → both agents in parallel --
    graph.add_edge("clause_extractor", "risk_assessor")
    graph.add_edge("clause_extractor", "missing_clause_checker")

    # -- Fan-in: both parallel agents → summariser --
    graph.add_edge("risk_assessor", "summariser")
    graph.add_edge("missing_clause_checker", "summariser")

    # -- Summariser → Reviewer --
    graph.add_edge("summariser", "reviewer")

    # -- Reviewer: 3-way routing --
    graph.add_conditional_edges(
        "reviewer",
        reviewer_routing,
        {
            "complete": END,
            "summariser": "summariser",         # revise summary only
            "risk_assessor": "risk_assessor",   # revise risk → then summariser
            "error_handler": "error_handler",
        },
    )

    graph.add_edge("error_handler", END)

    return graph


def compile_graph():
    graph = build_graph()
    return graph.compile()
