"""
LangGraph pipeline — Multi-Format Version with Tool Calling.

This version shows how the graph changes when the parser agent
uses tool calling instead of direct function calls.

Key difference: there's now a LOOP in the parser section:
  parser_agent → (tool call?) → parser_tools → parser_agent → (done?) → clause_extractor

This is the ReAct pattern: Reason → Act → Observe → Reason again.
"""

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from models.state import ContractAnalysisState
from agents.parser_with_tools import parser_agent, PARSER_TOOLS
from agents.clause_extractor import clause_extractor_agent
from agents.risk_assessor import risk_assessor_agent
from agents.summariser import summariser_agent


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


def error_handler(state: ContractAnalysisState) -> dict:
    error_msg = state.get("error_message", "Unknown error occurred.")
    print(f"[ERROR] Pipeline failed: {error_msg}")
    return {"current_step": "error"}


def build_graph() -> StateGraph:
    """
    Construct the LangGraph pipeline with tool calling.

    The key structural difference from the simple version:

    SIMPLE (direct calls):
        parser ──→ clause_extractor ──→ risk_assessor ──→ summariser ──→ END

    WITH TOOL CALLING (ReAct loop + contract validation + RAG):

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
           │       ▼
           │  risk_assessor ◄─── ChromaDB (CUAD benchmarks)
           │       │             11K clauses from 510 SEC filings
           │       ▼             semantic search per clause
           │  summariser
           │       │
           │       ▼
           │    complete ──→ END
           │
           ▼
        ┌──────────────────────┐
        │  error_handler       │──→ END
        │  "Not a contract"    │
        └──────────────────────┘

    This loop means the parser can:
    1. Try parse_pdf → get little text → try ocr_scanned_document
    2. Extract text → validate it's a contract → proceed
    3. Detect non-contract documents → short-circuit to error handler

    The risk assessor queries a ChromaDB vector store containing 11,129
    benchmark clauses from the CUAD dataset (SEC EDGAR filings) to ground
    its assessment in real-world contract standards.
    """

    graph = StateGraph(ContractAnalysisState)

    # -- Nodes --
    graph.add_node("parser_agent", parser_agent)
    graph.add_node("parser_tools", ToolNode(tools=PARSER_TOOLS))
    graph.add_node("clause_extractor", clause_extractor_agent)
    graph.add_node("risk_assessor", risk_assessor_agent)
    graph.add_node("summariser", summariser_agent)
    graph.add_node("error_handler", error_handler)

    # -- Entry point --
    graph.set_entry_point("parser_agent")

    # -- Parser agent: ReAct loop --
    # After the parser agent runs, check if it emitted tool calls
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

    # -- Rest of pipeline (same as before) --
    graph.add_conditional_edges(
        "clause_extractor",
        route_or_error("risk_assessor"),
        {
            "risk_assessor": "risk_assessor",
            "error_handler": "error_handler",
        },
    )

    graph.add_conditional_edges(
        "risk_assessor",
        route_or_error("summariser"),
        {
            "summariser": "summariser",
            "error_handler": "error_handler",
        },
    )

    graph.add_conditional_edges(
        "summariser",
        route_or_error("complete"),
        {
            "complete": END,
            "error_handler": "error_handler",
        },
    )

    graph.add_edge("error_handler", END)

    return graph


def compile_graph():
    graph = build_graph()
    return graph.compile()
