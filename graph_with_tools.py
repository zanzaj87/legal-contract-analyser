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


def route_after_step(state: ContractAnalysisState) -> str:
    """Route to the next agent based on current_step."""
    step = state.get("current_step", "error")
    routing = {
        "extract": "clause_extractor",
        "assess_risk": "risk_assessor",
        "summarise": "summariser",
        "complete": "complete",
        "error": "error_handler",
    }
    return routing.get(step, "error_handler")


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

    WITH TOOL CALLING (ReAct loop):
        ┌──────────────────────┐
        │  parser_agent        │◄─────────┐
        │  (LLM reasons,      │          │
        │   emits tool_calls)  │          │ loop back with
        └──────┬───────────────┘          │ tool results
               │                          │
               ▼                          │
        ┌──────────────────────┐          │
        │  tools_condition     │          │
        │  (has tool calls?)   │          │
        └──┬───────────┬───────┘          │
           │           │                  │
      no tools     has tools              │
           │           │                  │
           ▼           ▼                  │
     clause_       ┌───────────────┐      │
     extractor     │  parser_tools │      │
                   │  (executes    │──────┘
                   │   the tool)   │
                   └───────────────┘

    This loop means the parser can:
    1. Try parse_pdf → get little text → try ocr_scanned_document
    2. Extract text → validate it's a contract → proceed
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
        tools_condition,  # built-in: checks for tool_calls in the AIMessage
        {
            # If the LLM wants to call a tool → execute it
            "tools": "parser_tools",
            # If no tool calls → LLM is done reasoning, move to extraction
            END: "clause_extractor",
        },
    )

    # After tools execute, loop back to the parser agent
    # so the LLM can see the results and decide what to do next
    graph.add_edge("parser_tools", "parser_agent")

    # -- Rest of pipeline (same as before) --
    graph.add_conditional_edges(
        "clause_extractor",
        route_after_step,
        {
            "risk_assessor": "risk_assessor",
            "error_handler": "error_handler",
        },
    )

    graph.add_conditional_edges(
        "risk_assessor",
        route_after_step,
        {
            "summariser": "summariser",
            "error_handler": "error_handler",
        },
    )

    graph.add_conditional_edges(
        "summariser",
        route_after_step,
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
