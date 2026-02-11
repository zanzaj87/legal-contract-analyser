"""
Streamlit UI for the Legal Contract Analyser.
Run with: streamlit run app.py
"""

import streamlit as st
import tempfile
import os
from pathlib import Path

from graph import compile_graph
from models.schemas import ClauseExtractionResult, RiskAssessmentResult

from dotenv import load_dotenv

load_dotenv(override=True)

st.set_page_config(
    page_title="Legal Contract Analyser",
    page_icon="📑",
    layout="wide",
)

st.title("📑 Legal Contract Analyser")
st.markdown(
    "Upload a contract PDF to extract clauses, assess risks, "
    "and generate an executive summary using a **multi-agent AI pipeline**."
)

# Sidebar — architecture explanation (good for interview demos)
with st.sidebar:
    st.header("🏗️ Architecture")
    st.markdown(
        """
        This tool uses a **multi-agent system** built with [LangGraph](https://langchain-ai.github.io/langgraph/):

        1. **Parser Agent** — Extracts text from PDF, validates it's a contract
        2. **Clause Extractor** — Identifies key legal clauses using structured LLM output
        3. **Risk Assessor** — Evaluates each clause for legal risk
        4. **Summariser** — Produces an executive summary

        Each agent is a **node** in a directed graph with **conditional routing**.
        If any step fails, the pipeline routes to an error handler.
        """
    )
    st.divider()
    st.markdown("**Tech Stack:** LangGraph · GPT-4o · PyMuPDF · Pydantic · Streamlit")


# File upload
uploaded_file = st.file_uploader(
    "Upload a contract PDF",
    type=["pdf"],
    help="Supported: PDF files up to 50 pages",
)

if uploaded_file is not None:
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.success(f"Uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

    if st.button("🔍 Analyse Contract", type="primary", use_container_width=True):
        # Progress tracking
        progress = st.progress(0, text="Initialising pipeline...")

        # Compile graph
        app = compile_graph()

        initial_state = {
            "pdf_path": tmp_path,
            "parsed_text": None,
            "document_metadata": None,
            "extraction_result": None,
            "risk_result": None,
            "executive_summary": None,
            "current_step": "parse",
            "error_message": None,
        }

        # Stream through the pipeline
        step_labels = {
            "parser": ("Parsing PDF...", 0.2),
            "clause_extractor": ("Extracting clauses...", 0.4),
            "risk_assessor": ("Assessing risks...", 0.7),
            "summariser": ("Generating summary...", 0.9),
        }

        final_state = None
        for step_output in app.stream(initial_state):
            node_name = list(step_output.keys())[0]
            if node_name in step_labels:
                label, pct = step_labels[node_name]
                progress.progress(pct, text=f"✓ {label}")
            final_state = step_output.get(node_name, {})

        # Get complete final state
        final_state = app.invoke(initial_state)
        progress.progress(1.0, text="✓ Analysis complete!")

        # Check for errors
        if final_state.get("current_step") == "error":
            st.error(f"Analysis failed: {final_state.get('error_message')}")
        else:
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(
                ["📋 Executive Summary", "📄 Extracted Clauses", "⚠️ Risk Assessment"]
            )

            # Tab 1: Executive Summary
            with tab1:
                summary = final_state.get("executive_summary", "")
                if summary:
                    st.markdown(summary)

            # Tab 2: Extracted Clauses
            with tab2:
                extraction: ClauseExtractionResult = final_state.get("extraction_result")
                if extraction:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Contract Type", extraction.contract_type)
                    col2.metric("Parties", len(extraction.parties))
                    col3.metric("Clauses Found", len(extraction.clauses))

                    if extraction.parties:
                        st.markdown(f"**Parties:** {', '.join(extraction.parties)}")
                    if extraction.effective_date:
                        st.markdown(f"**Effective Date:** {extraction.effective_date}")

                    st.divider()

                    for clause in extraction.clauses:
                        with st.expander(
                            f"**{clause.clause_type.replace('_', ' ').title()}** "
                            f"— {clause.title} ({clause.section_reference})"
                        ):
                            st.text(clause.text)

            # Tab 3: Risk Assessment
            with tab3:
                risk: RiskAssessmentResult = final_state.get("risk_result")
                if risk:
                    # Overall risk badge
                    risk_colors = {
                        "low": "🟢",
                        "medium": "🟡",
                        "high": "🔴",
                    }
                    st.markdown(
                        f"### Overall Risk: {risk_colors.get(risk.overall_risk, '⚪')} "
                        f"{risk.overall_risk.upper()}"
                    )
                    st.markdown(f"*{risk.summary_of_concerns}*")

                    if risk.missing_clauses:
                        st.warning(
                            f"**Missing Clauses:** {', '.join(risk.missing_clauses)}"
                        )

                    st.divider()

                    for assessment in risk.clause_assessments:
                        emoji = risk_colors.get(assessment.risk_level, "⚪")
                        with st.expander(
                            f"{emoji} **{assessment.clause_type.replace('_', ' ').title()}** "
                            f"({assessment.section_reference}) — {assessment.risk_level.upper()}"
                        ):
                            st.markdown(f"**Assessment:** {assessment.risk_reasoning}")
                            if assessment.key_concerns:
                                st.markdown("**Concerns:**")
                                for concern in assessment.key_concerns:
                                    st.markdown(f"- {concern}")
                            st.info(f"**Recommendation:** {assessment.recommendation}")

        # Cleanup temp file
        os.unlink(tmp_path)
