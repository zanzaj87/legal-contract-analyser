"""
CLI entry point for the Legal Contract Analyser.
Usage: python main.py --contract path/to/contract.pdf
"""

import argparse
import json
from graph import compile_graph
from models.schemas import ClauseExtractionResult, RiskAssessmentResult

from dotenv import load_dotenv

load_dotenv(override=True)


def print_results(final_state: dict) -> None:
    """Pretty-print the analysis results."""

    if final_state.get("current_step") == "error":
        print(f"\n{'='*60}")
        print(f"ANALYSIS FAILED")
        print(f"{'='*60}")
        print(f"Error: {final_state.get('error_message', 'Unknown')}")
        return

    print(f"\n{'='*60}")
    print(f" LEGAL CONTRACT ANALYSIS REPORT")
    print(f"{'='*60}")

    # Document metadata
    meta = final_state.get("document_metadata", {})
    print(f"\nFile: {meta.get('filename', 'N/A')}")
    print(f"Pages: {meta.get('page_count', 'N/A')}")

    # Extraction results
    extraction: ClauseExtractionResult = final_state.get("extraction_result")
    if extraction:
        print(f"\nContract Type: {extraction.contract_type}")
        print(f"Parties: {', '.join(extraction.parties)}")
        print(f"Effective Date: {extraction.effective_date or 'Not specified'}")
        print(f"Clauses Found: {len(extraction.clauses)}")

        print(f"\n{'─'*40}")
        print("EXTRACTED CLAUSES")
        print(f"{'─'*40}")
        for clause in extraction.clauses:
            print(f"\n  [{clause.clause_type.upper()}] {clause.title}")
            print(f"  Section: {clause.section_reference}")
            print(f"  Text: {clause.text[:200]}...")

    # Risk results
    risk: RiskAssessmentResult = final_state.get("risk_result")
    if risk:
        print(f"\n{'─'*40}")
        print(f"RISK ASSESSMENT — Overall: {risk.overall_risk.upper()}")
        print(f"{'─'*40}")

        for assessment in risk.clause_assessments:
            emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
                assessment.risk_level, "⚪"
            )
            print(
                f"\n  {emoji} {assessment.clause_type} ({assessment.section_reference})"
                f" — {assessment.risk_level.upper()}"
            )
            print(f"     {assessment.risk_reasoning}")
            if assessment.key_concerns:
                for concern in assessment.key_concerns:
                    print(f"     • {concern}")
            print(f"     → {assessment.recommendation}")

        if risk.missing_clauses:
            print(f"\n  ⚠️  Missing Clauses: {', '.join(risk.missing_clauses)}")

    # Executive summary
    summary = final_state.get("executive_summary")
    if summary:
        print(f"\n{'─'*40}")
        print("EXECUTIVE SUMMARY")
        print(f"{'─'*40}")
        print(f"\n{summary}")

    print(f"\n{'='*60}")
    print("Analysis complete.")


def main():
    parser = argparse.ArgumentParser(description="Legal Contract Analyser")
    parser.add_argument(
        "--contract",
        type=str,
        required=True,
        help="Path to the contract PDF file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print intermediate state after each agent",
    )
    args = parser.parse_args()

    print(f"Analysing contract: {args.contract}")
    print("Building agent pipeline...")

    # Compile the LangGraph
    app = compile_graph()

    # Run the pipeline
    initial_state = {
        "pdf_path": args.contract,
        "parsed_text": None,
        "document_metadata": None,
        "extraction_result": None,
        "risk_result": None,
        "executive_summary": None,
        "current_step": "parse",
        "error_message": None,
    }

    print("Running analysis pipeline...\n")

    if args.verbose:
        # Stream mode: see each step
        for step_output in app.stream(initial_state):
            node_name = list(step_output.keys())[0]
            print(f"  ✓ Completed: {node_name}")
        # Get final state
        final_state = app.invoke(initial_state)
    else:
        final_state = app.invoke(initial_state)

    print_results(final_state)


if __name__ == "__main__":
    main()
