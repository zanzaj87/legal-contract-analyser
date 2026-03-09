# Legal Contract Analyser — Multi-Agent System

A multi-agent system built with **LangGraph** that analyses legal contracts by extracting clauses, assessing risks against real-world benchmarks, identifying missing protections, and producing executive summaries with an automated quality review loop.

Built as a portfolio demonstration of production-grade multi-agent architectures applied to the legal domain.

## Live Demo

**Try it out:** [legal-contract-analyser-zanzaj87.streamlit.app](https://legal-contract-analyser-zanzaj87.streamlit.app/)

## Architecture Overview

The system has evolved through three versions, each demonstrating progressively more advanced agent patterns.

### V1: Sequential Pipeline (`graph.py`)

Each agent is called deterministically in a fixed order. The parser calls `parse_pdf()` directly, no LLM decision-making.

```
parser ──→ clause_extractor ──→ risk_assessor ──→ summariser ──→ END
```

**When to use:** Input is always a PDF with selectable text. Simple, fast, easy to debug.

### V2: Tool-Calling Pipeline (intermediate)

The parser agent uses **LLM tool calling** (ReAct pattern) to dynamically select the right extraction method based on file type. Introduces a loop where the LLM reasons, picks a tool, sees the result, and decides what to do next.

**When to use:** Input could be PDF, DOCX, or scanned image. The LLM decides which parsing tool to call.

### V3: Full Production Pipeline (`graph_with_tools.py`) - Current

The complete architecture with five key patterns:

1. **ReAct tool calling**: Parser agent dynamically selects parsing tools
2. **Contract validation**: Early termination for non-contract documents
3. **RAG-enhanced assessment**: Risk assessor queries 11K benchmark clauses from SEC filings
4. **Parallel fan-out/fan-in**: Risk assessor and missing clause checker run simultaneously
5. **Reviewer feedback loop**: Automated quality check with targeted revision

```
    ┌──────────────────────┐
    │  parser_agent        │◄─────────┐
    │  (ReAct tool calling)│          │ loop back with
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
       │       │   └───────────────┘
       │       │
       │       ▼ (fan-out: parallel execution)
       │       ┌─────────────────────────────────┐
       │       │                                 │
       │       ▼                                 ▼
       │  risk_assessor ◄── ChromaDB    missing_clause_checker ◄── ChromaDB
       │  (assess clauses)   (CUAD)     (identify gaps)              (CUAD)
       │       │                                 │
       │       └────────────┬────────────────────┘
       │                    ▼ (fan-in)
       │               summariser ◄──────────────────────┐
       │                    │                            │
       │                    ▼                            │
       │               reviewer                          │
       │                    │                            │
       │            ┌───────┼───────┐                    │
       │         approve  revise  revise                 │
       │            │     summary  risk                  │
       │            ▼       │       │                    │
       │          END       │       ▼                    │
       │                    │  risk_assessor             │
       │                    │       │                    │
       │                    └───────┴────────────────────┘
       │                    (max 1 revision, then forced approve)
       ▼
    ┌──────────────────────┐
    │  error_handler       │──→ END
    └──────────────────────┘
```

## Agents

| Agent | Model | Purpose |
|-------|-------|---------|
| **Parser** | `gpt-5-mini` | Extracts text via tool calling (PDF/DOCX/OCR), validates it's a contract |
| **Clause Extractor** | `gpt-5.2` | Identifies key legal clauses using structured output (Pydantic) |
| **Risk Assessor** | `gpt-5.2` | Evaluates each clause against RAG benchmarks, assigns risk levels |
| **Missing Clause Checker** | `gpt-5-mini` | Identifies expected but absent clauses using CUAD benchmark data |
| **Summariser** | `gpt-5.2` | Merges both parallel outputs into an executive summary |
| **Reviewer** | `gpt-5-mini` | Checks consistency across all outputs, can request targeted revisions |

## Key Features

### RAG-Enhanced Risk Assessment

The risk assessor queries a **ChromaDB** vector store containing **11,129 benchmark clauses** from the [CUAD dataset](https://www.atticusprojectai.org/cuad) - 510 real commercial contracts filed with the SEC. For each extracted clause, it retrieves semantically similar benchmarks and compares the uploaded contract against real-world market practice.

```
Uploaded clause: "Liability capped at 12 months' fees..."
    ↓ semantic search (OpenAI text-embedding-3-small)
    ↓
ChromaDB returns 3 similar clauses from SEC filings
    ↓
LLM: "This cap is consistent with 3/3 benchmark clauses from similar contracts"
```

### Parallel Agent Execution

After clause extraction, two analysis agents run **simultaneously**:
- **Risk Assessor**: evaluates existing clauses (with RAG benchmarks)
- **Missing Clause Checker**: identifies gaps (using CUAD category statistics)

LangGraph's fan-out/fan-in pattern reduces total latency by ~40% compared to sequential execution.

### Reviewer Feedback Loop

The reviewer agent checks consistency across all outputs:
- Are all HIGH-risk findings in the executive summary?
- Do missing clause counts match between the analysis and summary?
- Are recommendations specific and actionable?

If issues are found, it sends targeted revision instructions back to the summariser or risk assessor. Limited to 1 revision to prevent infinite loops.

### Contract Validation

The parser agent validates whether the uploaded document is actually a legal contract. Non-contracts (e.g., museum tickets, receipts) are rejected early, saving tokens and preventing nonsensical analysis.

## Project Structure

```
legal-contract-analyser/
├── agents/
│   ├── prompts.py              # All prompt templates
│   ├── parser.py               # V1: Direct function call parser
│   ├── parser_with_tools.py    # V2/V3: Tool-calling parser (ReAct)
│   ├── clause_extractor.py     # Structured clause extraction
│   ├── risk_assessor.py        # RAG-enhanced risk assessment
│   ├── missing_clause_checker.py  # Missing clause identification
│   ├── summariser.py           # Executive summary (merges parallel outputs)
│   └── reviewer.py             # Quality review with feedback loop
├── models/
│   ├── state.py                # ContractAnalysisState (shared graph state)
│   └── schemas.py              # Pydantic models (extraction, risk, missing, review)
├── rag/
│   ├── __init__.py
│   ├── shared.py               # Singleton retriever instance
│   ├── ingest.py               # CUAD dataset → ChromaDB ingestion pipeline
│   ├── vectorstore.py          # ClauseRetriever (semantic search interface)
│   └── chroma_db/              # Persisted vector store (11K embeddings)
├── tools/
│   └── pdf_parser.py           # PyMuPDF / Tesseract extraction tools
├── utils/
│   └── llm.py                  # Centralised LLM configuration
├── graph.py                    # V1: Sequential pipeline
├── graph_with_tools.py         # V3: Full pipeline (tools + parallel + reviewer)
├── app.py                      # Streamlit UI
├── main.py                     # CLI entry point
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
# Edit .env and add your OPENAI_API_KEY

# 4. Build the RAG vector store (one-time, ~$0.08 in embedding costs)
python rag/ingest.py

# 5. Run
streamlit run app.py
```

## Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=sk-...

# Optional: LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
LANGSMITH_PROJECT=legal-contract-analyser
```

## RAG Ingestion

The vector store needs to be built once before first use:

```bash
# Preview the dataset (no API calls)
python rag/ingest.py --stats

# Full ingestion: download CUAD → embed → store in ChromaDB
python rag/ingest.py
```

This downloads the [CUAD v1 clause classification dataset](https://huggingface.co/datasets/dvgodoy/CUAD_v1_Contract_Understanding_clause_classification) (13,155 labelled clauses from 510 SEC EDGAR contracts), processes them into 11,129 benchmark clauses across 41 categories, and embeds them using OpenAI `text-embedding-3-small`.

## Tech Stack

- **LangGraph** - Agent orchestration, parallel execution, conditional routing
- **LangChain** - LLM integration, tool calling, structured output
- **ChromaDB** - Vector store for RAG benchmark clauses
- **OpenAI** - GPT-5.2 / GPT-5-mini (models), text-embedding-3-small (embeddings)
- **CUAD Dataset** - 510 SEC EDGAR contracts, 41 clause categories (CC BY 4.0)
- **LangSmith** - Observability and tracing (including RAG retrieval spans)
- **PyMuPDF** - PDF text extraction
- **Pydantic** - Structured LLM output schemas
- **Streamlit** - Web UI

## Observability

With LangSmith tracing enabled, every pipeline run produces a detailed trace showing:
- ReAct loop iterations (parser tool calls and responses)
- RAG retrieval spans (query text, benchmark results, distances)
- Parallel agent execution timing
- Reviewer decisions and revision instructions
- Token usage and cost per agent

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pure semantic search (no category mapping) for RAG | Embedding model finds relevant benchmarks regardless of label taxonomy differences |
| Parallel fan-out via simple edges (not conditional) | Two `add_edge()` calls from one node is the simplest LangGraph fan-out pattern |
| Reviewer uses `gpt-5-mini` | Consistency checking is a lighter task than risk assessment; saves cost |
| Max 1 revision loop | Diminishing returns after one correction; prevents runaway costs |
| Shared singleton retriever | Avoids ChromaDB file locking issues when parallel agents access the vector store |
| State reducers for `current_step` and `error_message` | Required for parallel agents writing to the same state fields simultaneously |

## Licence

This project is for portfolio/demonstration purposes. The CUAD dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).