"""
Missing Clause Checker Agent.
Identifies clauses that are expected for the contract type but absent.
Runs in parallel with the risk assessor after clause extraction.

Uses RAG to determine what clause types typically appear in similar contracts
from the CUAD dataset.
"""

from models.state import ContractAnalysisState
from models.schemas import MissingClauseResult, ClauseExtractionResult
from utils.llm import get_llm
from agents.prompts import MISSING_CLAUSE_CHECKER_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage
from rag.shared import get_shared_retriever


# ─── RAG Retriever (lazy-loaded) ─────────────────────────────────────────────


def _build_rag_context_for_missing_clauses(extraction: ClauseExtractionResult) -> str:
    """
    Query the vector store to find what clause categories typically appear
    in this type of contract, to help identify gaps.
    """
    retriever = get_shared_retriever()
    if not retriever:
        return ""

    try:
        # Get the distribution of categories in the vector store
        stats = retriever.get_category_stats()

        # For each clause type NOT found in the contract, check if the
        # vector store has benchmarks — if it does, it's likely expected
        found_types = {c.clause_type.lower() for c in extraction.clauses}

        context_parts = [
            "\n\n" + "=" * 60,
            "BENCHMARK DATA: Clause categories from 510 SEC EDGAR filings (CUAD dataset)",
            "Categories with more benchmarks are more commonly found in commercial contracts.",
            "=" * 60,
            f"\nClause categories in the benchmark database ({len(stats)} total):\n",
        ]

        for category, count in stats.items():
            context_parts.append(f"  {category}: {count} examples")

        context_parts.append(
            f"\n\nThe uploaded contract ({extraction.contract_type}) contains "
            f"these clause types: {', '.join(c.clause_type for c in extraction.clauses)}"
        )
        context_parts.append(
            "\nUse the benchmark frequency data above to assess which missing "
            "clause types are commonly expected in similar contracts."
        )

        return "\n".join(context_parts)

    except Exception as e:
        print(f"[RAG] Error building missing clause context: {e}")
        return ""


def _format_extraction_for_checker(extraction: ClauseExtractionResult) -> str:
    """Format the extraction result for the missing clause checker."""
    parts = [
        f"Contract Type: {extraction.contract_type}",
        f"Parties: {', '.join(extraction.parties)}",
    ]
    if extraction.effective_date:
        parts.append(f"Effective Date: {extraction.effective_date}")

    parts.append(f"\nClauses FOUND in this contract ({len(extraction.clauses)}):")
    parts.append("-" * 60)

    for i, clause in enumerate(extraction.clauses, 1):
        parts.append(
            f"  [{i}] {clause.clause_type} — {clause.title} ({clause.section_reference})"
        )

    return "\n".join(parts)


def missing_clause_checker_agent(state: ContractAnalysisState) -> dict:
    """
    Node: Identify missing clauses expected for this contract type.

    Runs in PARALLEL with the risk assessor after clause extraction.
    Uses RAG to understand what clause types are standard in similar contracts.

    Reads: extraction_result
    Writes: missing_clause_result, current_step
    """
    extraction = state.get("extraction_result")

    if not extraction:
        return {
            "current_step": "error",
            "error_message": "No extraction results available for missing clause check.",
        }

    try:
        llm = get_llm(model="gpt-5-mini", temperature=0.0)
        structured_llm = llm.with_structured_output(MissingClauseResult)

        # Format what was found
        extraction_text = _format_extraction_for_checker(extraction)

        # Build RAG context about typical clause categories
        rag_context = _build_rag_context_for_missing_clauses(extraction)

        human_message = (
            f"Analyse the following contract for missing clauses:\n\n"
            f"{extraction_text}"
        )

        if rag_context:
            human_message += (
                f"\n\n{rag_context}"
                f"\n\nIMPORTANT: Use the benchmark frequency data to support your "
                f"assessment of which clauses are typically expected. Categories with "
                f"hundreds of examples in the benchmark are very common in commercial "
                f"contracts and their absence is more notable."
            )

        result: MissingClauseResult = structured_llm.invoke(
            [
                SystemMessage(content=MISSING_CLAUSE_CHECKER_SYSTEM_PROMPT),
                HumanMessage(content=human_message),
            ]
        )

        return {
            "missing_clause_result": result,
            "current_step": "summarise",
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "current_step": "error",
            "error_message": f"Missing clause check failed: {str(e)}",
        }