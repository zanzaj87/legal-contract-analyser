"""
LangGraph pipeline definition.
This is the core of the multi-agent system — it wires agents together
as nodes in a directed graph with conditional routing.
"""

from langgraph.graph import StateGraph, END
from models.state import ContractAnalysisState
from agents.parser import parser_agent
from agents.clause_extractor import clause_extractor_agent
from agents.risk_assessor import risk_assessor_agent
from agents.summariser import summariser_agent


def route_after_step(state: ContractAnalysisState) -> str:
    """
    Conditional edge: decide which node to go to next based on current_step.

    This is the 'supervisor' logic — it routes the workflow based on
    what the previous agent wrote to state.
    """
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
    """Terminal node for error states."""
    error_msg = state.get("error_message", "Unknown error occurred.")
    print(f"[ERROR] Pipeline failed: {error_msg}")
    return {"current_step": "error"}


def build_graph() -> StateGraph:
    """
    Construct the LangGraph pipeline.

    Flow:
        parser → (conditional) → clause_extractor → risk_assessor → summariser → END
                                                                                ↘ error → END
    """
    graph = StateGraph(ContractAnalysisState)

    # -- Add nodes (each agent is a node) --
    graph.add_node("parser", parser_agent)
    graph.add_node("clause_extractor", clause_extractor_agent)
    graph.add_node("risk_assessor", risk_assessor_agent)
    graph.add_node("summariser", summariser_agent)
    graph.add_node("error_handler", error_handler)

    # -- Set entry point --
    graph.set_entry_point("parser")

    # -- Add conditional edges (the routing logic) --
    # After the parser, route based on what it wrote to current_step
    graph.add_conditional_edges(
        "parser",
        route_after_step,
        {
            "clause_extractor": "clause_extractor",
            "error_handler": "error_handler",
        },
    )

    # After clause extraction, route to risk assessor or error
    graph.add_conditional_edges(
        "clause_extractor",
        route_after_step,
        {
            "risk_assessor": "risk_assessor",
            "error_handler": "error_handler",
        },
    )

    # After risk assessment, route to summariser or error
    graph.add_conditional_edges(
        "risk_assessor",
        route_after_step,
        {
            "summariser": "summariser",
            "error_handler": "error_handler",
        },
    )

    # After summariser, go to END or error
    graph.add_conditional_edges(
        "summariser",
        route_after_step,
        {
            "complete": END,
            "error_handler": "error_handler",
        },
    )

    # Error handler always goes to END
    graph.add_edge("error_handler", END)

    return graph


def compile_graph():
    """Build and compile the graph, ready for invocation."""
    graph = build_graph()
    return graph.compile()
