"""
LLM configuration.
Centralised so you can swap models easily for the whole pipeline.
"""

import os
from dotenv import load_dotenv

load_dotenv(override=True)


def get_llm(model: str = "gpt-5.1", temperature: float = 0.0):
    """
    Return a LangChain chat model.

    Supports:
        - OpenAI
        - Anthropic (uncomment langchain-anthropic in requirements.txt)
    """
    if model.startswith("claude"):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature)
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)
