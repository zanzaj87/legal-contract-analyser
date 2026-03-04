"""
Vector Store query interface for the Legal Contract Analyser.

Provides retrieval functions that the risk assessor agent uses
to find benchmark clauses from the CUAD dataset.

Usage:
    from rag.vectorstore import ClauseRetriever

    retriever = ClauseRetriever()

    # Find similar termination clauses
    results = retriever.find_similar_clauses(
        clause_text="Either party may terminate upon 30 days notice...",
        category="Termination For Convenience",
        n_results=5,
    )

    # Find all clauses of a specific category
    results = retriever.get_clauses_by_category("Governing Law", n_results=10)

    # Find benchmark clauses for risk comparison
    context = retriever.get_risk_assessment_context(
        clause_text="...",
        category="Cap On Liability",
    )
"""

import os
from pathlib import Path
from langsmith import traceable


try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

import chromadb
from openai import OpenAI


# ─── Configuration ───────────────────────────────────────────────────────────

CHROMA_DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "clause_benchmarks"
EMBEDDING_MODEL = "text-embedding-3-small"


# ─── Retriever Class ─────────────────────────────────────────────────────────

class ClauseRetriever:
    """
    Retrieves benchmark clauses from the CUAD ChromaDB vector store.

    Used by the risk assessor agent to compare uploaded contract clauses
    against real-world examples from SEC EDGAR filings.
    """

    def __init__(self, db_path: str | Path | None = None):
        db_path = Path(db_path) if db_path else CHROMA_DB_PATH

        if not db_path.exists():
            raise FileNotFoundError(
                f"ChromaDB not found at {db_path}. "
                f"Run 'python rag/ingest.py' first to build the vector store."
            )

        self.chroma_client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        print(f"[RAG] Loaded clause benchmark store: {self.collection.count():,} clauses")

    def _embed_query(self, text: str) -> list[float]:
        """Embed a query text using OpenAI."""
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    @traceable(name="RAG: Retrieve Benchmark Clauses", run_type="retriever")
    def find_similar_clauses(
        self,
        clause_text: str,
        category: str | None = None,
        n_results: int = 5,
    ) -> list[dict]:
        """
        Find similar benchmark clauses using semantic search.

        Args:
            clause_text: The clause from the uploaded contract to compare against.
            category: Optional — filter results to a specific clause category.
            n_results: Number of similar clauses to return.

        Returns:
            List of dicts with keys: clause_text, category, contract_type,
            source_file, distance (similarity score).
        """
        # Build the query text with category context (mirrors how we embedded)
        if category:
            query_text = f"[{category}]\n{clause_text}"
        else:
            query_text = clause_text

        query_embedding = self._embed_query(query_text)

        # Build optional category filter
        where_filter = None
        if category:
            where_filter = {"category": category}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "clause_text": results["documents"][0][i],
                "category": results["metadatas"][0][i].get("category", "Unknown"),
                "contract_type": results["metadatas"][0][i].get("contract_type", "Unknown"),
                "source_file": results["metadatas"][0][i].get("source_file", "Unknown"),
                "distance": results["distances"][0][i],
            })

        return formatted

    def get_clauses_by_category(
        self,
        category: str,
        n_results: int = 10,
    ) -> list[dict]:
        """
        Get benchmark clauses for a specific category.

        Useful for understanding what standard clauses in a category look like.
        """
        results = self.collection.get(
            where={"category": category},
            limit=n_results,
            include=["documents", "metadatas"],
        )

        formatted = []
        for i in range(len(results["ids"])):
            formatted.append({
                "clause_text": results["documents"][i],
                "category": results["metadatas"][i].get("category", "Unknown"),
                "contract_type": results["metadatas"][i].get("contract_type", "Unknown"),
                "source_file": results["metadatas"][i].get("source_file", "Unknown"),
            })

        return formatted

    def get_risk_assessment_context(
        self,
        clause_text: str,
        category: str,
        n_similar: int = 5,
    ) -> str:
        """
        Build a context string for the risk assessor agent.

        Returns a formatted string containing:
        1. Similar benchmark clauses for comparison
        2. Summary statistics about the category

        This gets injected into the risk assessor's prompt alongside
        the clause being analysed.
        """
        # Get similar clauses
        similar = self.find_similar_clauses(
            clause_text=clause_text,
            category=category,
            n_results=n_similar,
        )

        # Get total count for this category
        category_results = self.collection.get(
            where={"category": category},
            include=["metadatas"],
        )
        total_in_category = len(category_results["ids"])

        # Build context string
        lines = [
            f"=== RAG BENCHMARK CONTEXT: {category} ===",
            f"Total benchmark clauses in this category: {total_in_category}",
            f"Showing {len(similar)} most similar clauses from SEC EDGAR filings:\n",
        ]

        for i, clause in enumerate(similar, 1):
            lines.append(f"--- Benchmark {i} (distance: {clause['distance']:.4f}) ---")
            lines.append(f"Contract type: {clause['contract_type']}")
            lines.append(f"Source: {clause['source_file'][:60]}...")
            # Truncate very long clauses for context window efficiency
            text = clause["clause_text"]
            if len(text) > 500:
                text = text[:500] + "... [truncated]"
            lines.append(f"Text: {text}")
            lines.append("")

        lines.append("=== END BENCHMARK CONTEXT ===")
        return "\n".join(lines)

    def get_category_stats(self) -> dict[str, int]:
        """Get count of clauses per category in the store."""
        all_items = self.collection.get(include=["metadatas"])
        counts = {}
        for meta in all_items["metadatas"]:
            cat = meta.get("category", "Unknown")
            counts[cat] = counts.get(cat, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))
