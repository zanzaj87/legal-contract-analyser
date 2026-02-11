# Legal Contract Analyser — Multi-Agent Demo

A multi-agent system built with LangGraph that analyses legal contracts by extracting clauses, assessing risks, and producing executive summaries.

## Architecture

```
User uploads contract PDF
        │
        ▼
┌─────────────────────┐
│  SUPERVISOR AGENT    │  Orchestrates the workflow
│  (Conditional Router)│
└─────────┬───────────┘
          │
   ┌──────┼──────┬───────────┐
   ▼      ▼      ▼           ▼
┌──────┐┌───────┐┌─────────┐┌──────────┐
│Parser││Clause ││  Risk   ││Summariser│
│Agent ││Extract││Assessor ││  Agent   │
│      ││  or   ││  Agent  ││          │
└──────┘└───────┘└─────────┘└──────────┘
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file:
```
OPENAI_API_KEY=sk-...          # or use Anthropic
# ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
# CLI mode
python main.py --contract sample_contracts/sample_nda.pdf

# Streamlit UI
streamlit run app.py
```

## Sample Contracts
Place PDF contracts in `sample_contracts/` directory.
Free sources:
- https://www.lawinsider.com/clauses
- SEC EDGAR filings
- Google "sample NDA PDF" / "sample SaaS agreement PDF"
