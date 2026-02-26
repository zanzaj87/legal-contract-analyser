# Legal Contract Analyser — Multi-Agent Demo

A multi-agent system built with **LangGraph** that analyses legal contracts by extracting clauses, assessing risks, and producing executive summaries.

Built as a demonstration of multi-agent architectures applied to the legal domain.

## Architectures

The project includes two pipeline versions to demonstrate different agent design patterns.

### V1: Sequential Pipeline (`graph.py`)

Each agent is called deterministically in a fixed order. The parser calls `parse_pdf()` directly as a Python function — no LLM decision-making needed.

```
__start__
    │
    ▼
┌────────┐     ┌─────────────────┐     ┌───────────────┐     ┌────────────┐
│ Parser │ ──▶ │ Clause Extractor│ ──▶ │ Risk Assessor │ ──▶ │ Summariser │ ──▶ complete
└────────┘     └─────────────────┘     └───────────────┘     └────────────┘
    │                  │                       │                    │
    └──────────────────┴───────────────────────┴────────────────────┘
                                   │
                            error_handler ──▶ __end__
```

**When to use:** The input is always a PDF with selectable text. Simple, fast, easy to debug.

### V2: Tool-Calling Pipeline (`graph_with_tools.py`)

The parser agent uses **LLM tool calling** to dynamically select the right extraction method based on file type. This introduces a **ReAct loop** — the LLM reasons, picks a tool, sees the result, and decides what to do next.

```
__start__
    │
    ▼
┌──────────────┐    tools    ┌──────────────┐
│ Parser Agent │ ──────────▶ │ Parser Tools │
│  (LLM reasons│ ◀────────── │  (executes   │
│   & routes)  │   results   │   the tool)  │
└──────┬───────┘             └──────────────┘
       │ end                  Tools available:
       ▼                      • parse_pdf
┌─────────────────┐           • parse_docx
│ Clause Extractor│──────┐    • ocr_scanned_document
└────────┬────────┘      │
         ▼               │
┌───────────────┐        │
│ Risk Assessor │────────┤
└────────┬──────┘        │  on failure
         ▼               │
┌────────────┐           │
│ Summariser │───────────┤
└────────┬───┘           │
         │               ▼
         ▼         ┌───────────────┐
      complete     │ error_handler │
         │         └───────┬───────┘
         ▼                 ▼
             __end__
```

**When to use:** The input could be a PDF, DOCX, or scanned image. The LLM decides which parsing tool to call and can retry with a different tool if the first attempt returns poor results.

## Agents

| Agent | Model | Purpose |
|-------|-------|---------|
| **Parser** | `gpt-5-mini` | Extracts text from PDF, validates it's a contract |
| **Clause Extractor** | `gpt-5.2` | Identifies key legal clauses using structured output (Pydantic) |
| **Risk Assessor** | `gpt-5.2` | Evaluates each clause for risk level, flags concerns |
| **Summariser** | `gpt-5.2` | Produces an executive summary for non-technical stakeholders |

## Project Structure

```
legal-contract-analyser/
├── agents/
│   ├── prompts.py              # All prompt templates (separated for easy iteration)
│   ├── parser.py               # V1: Direct function call parser
│   ├── parser_with_tools.py    # V2: Tool-calling parser (ReAct pattern)
│   ├── clause_extractor.py     # Structured clause extraction
│   ├── risk_assessor.py        # Clause-by-clause risk assessment
│   └── summariser.py           # Executive summary generation
├── models/
│   ├── state.py                # ContractAnalysisState (shared graph state)
│   └── schemas.py              # Pydantic models for structured LLM output
├── tools/
│   └── pdf_parser.py           # PyMuPDF text extraction
├── utils/
│   └── llm.py                  # Centralised LLM configuration
├── graph.py                    # V1: Sequential pipeline (LangGraph)
├── graph_with_tools.py         # V2: Tool-calling pipeline (LangGraph)
├── main.py                     # CLI entry point
├── app.py                      # Streamlit UI
├── requirements.txt
└── .env.example
```

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
.\venv\Scripts\activate           # Windows (PowerShell)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create environment file
cp .env.example .env
# Edit .env and add your API key
```

## Environment Variables

Create a `.env` file:
```
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...   # uncomment if using Claude
```

## Test it out directly in Streamlit

https://legal-contract-analyser-zanzaj87.streamlit.app/

## Run locally

```bash
# CLI mode
python main.py --contract sample_contracts/test_contract.pdf

# Streamlit UI
streamlit run app.py
```

## Sample Contracts

Place PDF contracts in a `sample_contracts/` directory

## Tech Stack

- **LangGraph** — Agent orchestration and graph-based workflow
- **LangChain** — LLM integration and tool calling
- **GPT-5.2 / GPT-5-mini** — Language models (swappable via `utils/llm.py`)
- **PyMuPDF** — PDF text extraction
- **Pydantic** — Structured LLM output schemas
- **Streamlit** — Web UI
