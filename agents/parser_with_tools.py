"""
Parser Agent — Multi-Format Version.
Uses LLM tool calling to dynamically select the right extraction method
based on file type and content characteristics.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from models.state import ContractAnalysisState
from utils.llm import get_llm
from agents.prompts import MULTI_FORMAT_PARSER_PROMPT


from tools.tools import PARSER_TOOLS 
from pathlib import Path


def parser_agent(state: ContractAnalysisState) -> dict:
    """
    Node: Use tool calling to parse the uploaded document.

    The LLM decides which parsing tool to use based on the file type.
    This node returns an AIMessage with tool_calls — the ToolNode
    in the graph will execute the actual tool.
    """
    file_path = state["file_path"]
    file_ext = Path(file_path).suffix.lower()

    llm = get_llm(model="gpt-5-mini", temperature=0.0)
    llm_with_tools = llm.bind_tools(PARSER_TOOLS)

    # Check if we already have messages (i.e. we're in a loop iteration)
    existing_messages = state.get("messages", [])

    if existing_messages:
        # We're looping back — the LLM needs to see its previous
        # tool calls and the tool results to decide what to do next
        response = llm_with_tools.invoke(
        [SystemMessage(content=MULTI_FORMAT_PARSER_PROMPT)] + existing_messages
            )
    else:
        # First invocation — send the initial prompt
        response = llm_with_tools.invoke(
            [
                SystemMessage(content=MULTI_FORMAT_PARSER_PROMPT),
                HumanMessage(
                    content=f"Please extract text from this file: {file_path}\n"
                    f"File extension: {file_ext}"
                ),
            ]
        )

    # If the LLM is NOT calling any more tools, it means parsing is done.
    # Extract the parsed text from the tool results in messages.
    if not response.tool_calls:
        # Find the tool result message (contains the extracted text)
        parsed_text = None
        for msg in existing_messages:
            if hasattr(msg, 'type') and msg.type == 'tool':
                parsed_text = msg.content

        return {
            "messages": [response],
            "parsed_text": parsed_text,
            "current_step": "extract" if parsed_text else "error",
            "error_message": None if parsed_text else "Failed to extract text from document.",
        }

    return {"messages": [response]}