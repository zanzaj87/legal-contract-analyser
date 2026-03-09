"""
Risk Assessor Agent — RAG-Enhanced.
Evaluates extracted clauses for legal risk by comparing against
benchmark clauses from the CUAD dataset (SEC EDGAR filings).
"""

from models.state import ContractAnalysisState
from models.schemas import RiskAssessmentResult, ClauseExtractionResult
from utils.llm import get_llm
from agents.prompts import RISK_ASSESSOR_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage
from rag.shared import get_shared_retriever


# ─── Formatting ──────────────────────────────────────────────────────────────

def _format_clauses_for_assessment(extraction: ClauseExtractionResult) -> str:
    """Format extracted clauses into a readable string for the Risk Assessor."""
    parts = [
        f"Contract Type: {extraction.contract_type}",
        f"Parties: {', '.join(extraction.parties)}",
    ]
    if extraction.effective_date:
        parts.append(f"Effective Date: {extraction.effective_date}")

    parts.append(f"\nExtracted Clauses ({len(extraction.clauses)} found):")
    parts.append("-" * 60)

    for i, clause in enumerate(extraction.clauses, 1):
        parts.append(
            f"\n[{i}] {clause.clause_type.upper()} — {clause.title}"
            f"\n    Section: {clause.section_reference}"
            f"\n    Text: {clause.text}"
        )

    return "\n".join(parts)


def _build_rag_context(extraction: ClauseExtractionResult) -> str:
    """
    Query the vector store for benchmark clauses matching each extracted clause.
    Uses pure semantic search — no category mapping needed.
    """
    retriever = get_shared_retriever()
    print(f"[RAG DEBUG] Retriever returned: {retriever is not None}")
    print(f"[RAG DEBUG] Clauses to process: {len(extraction.clauses)}")
    if not retriever:
        return ""

    context_parts = [
        "\n\n" + "=" * 60,
        "BENCHMARK COMPARISON DATA (from CUAD — 510 SEC EDGAR filings)",
        "Use these real-world examples to calibrate your risk assessment.",
        "=" * 60,
    ]

    for clause in extraction.clauses:
        try:
            # Pure semantic search — let the embeddings find the best matches
            similar = retriever.find_similar_clauses(
                clause_text=clause.text,
                category=None,  # No filter — search across all categories
                n_results=2,
            )

            context_parts.append(f"\n[{clause.clause_type.upper()} — {clause.title}]")
            context_parts.append(f"Benchmarks found (top 3 by similarity):\n")

            for i, bench in enumerate(similar, 1):
                text = bench["clause_text"]
                if len(text) > 300:
                    text = text[:300] + "... [truncated]"
                context_parts.append(
                    f"  Benchmark {i} (distance: {bench['distance']:.4f}):"
                    f"\n    CUAD Category: {bench['category']}"
                    f"\n    Contract Type: {bench['contract_type']}"
                    f"\n    Text: {text}\n"
                )

        except Exception as e:
            print(f"[RAG] Error retrieving benchmarks for {clause.clause_type}: {e}")

    return "\n".join(context_parts)


# ─── Agent ───────────────────────────────────────────────────────────────────

def risk_assessor_agent(state: ContractAnalysisState) -> dict:
    """
    Node: Assess risk for each extracted clause.

    RAG-enhanced: queries the CUAD vector store for similar benchmark
    clauses from SEC EDGAR filings, then passes them to the LLM alongside
    the extracted clauses for grounded comparison.

    Reads: extraction_result
    Writes: risk_result, current_step
    """
    extraction = state.get("extraction_result")

    if not extraction:
        return {
            "current_step": "error",
            "error_message": "No extraction results available for risk assessment.",
        }

    try:
        llm = get_llm(model="gpt-5.2", temperature=0.0)
        structured_llm = llm.with_structured_output(RiskAssessmentResult)

        # Format the extracted clauses
        clauses_text = _format_clauses_for_assessment(extraction)

        # Build RAG benchmark context (empty string if store not available)
        rag_context = _build_rag_context(extraction)

        # Build the human message with optional RAG context
        human_message = (
            f"Assess the risk of the following contract clauses:\n\n"
            f"{clauses_text}"
        )

        if rag_context:
            human_message += (
                f"\n\n{rag_context}"
                f"\n\nIMPORTANT: Compare each clause against the benchmark examples above. "
                f"Identify what is standard practice, what deviates from typical terms, "
                f"and what protections are missing compared to similar contracts filed "
                f"with the SEC."
            )

        result: RiskAssessmentResult = structured_llm.invoke(
            [
                SystemMessage(content=RISK_ASSESSOR_SYSTEM_PROMPT),
                HumanMessage(content=human_message),
            ]
        )

        return {
            "risk_result": result,
            "current_step": "summarise",
        }

    except Exception as e:
        return {
            "current_step": "error",
            "error_message": f"Risk assessment failed: {str(e)}",
        }
