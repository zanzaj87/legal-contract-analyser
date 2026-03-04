"""
Quick test to verify the RAG pipeline works after ingestion.

Run this AFTER running: python rag/ingest.py

Usage:
    python rag/test_retrieval.py
"""

from rag.vectorstore import ClauseRetriever


def main():
    print("=" * 60)
    print("RAG Retrieval Test")
    print("=" * 60)

    # Load the retriever
    retriever = ClauseRetriever()

    # Show category stats
    print("\n📊 Category Statistics:")
    stats = retriever.get_category_stats()
    for cat, count in list(stats.items())[:10]:
        print(f"   {cat:40s} {count:5d}")
    print(f"   ... ({len(stats)} total categories)")

    # Test 1: Semantic search — find similar termination clauses
    print("\n" + "=" * 60)
    print("Test 1: Find similar Termination clauses")
    print("=" * 60)

    test_clause = (
        "Either party may terminate this Agreement at any time "
        "by providing thirty (30) days written notice to the other party."
    )
    results = retriever.find_similar_clauses(
        clause_text=test_clause,
        category="Termination For Convenience",
        n_results=3,
    )
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] Distance: {r['distance']:.4f}")
        print(f"      Type: {r['contract_type']}")
        print(f"      Text: {r['clause_text'][:200]}...")

    # Test 2: Risk assessment context
    print("\n" + "=" * 60)
    print("Test 2: Risk assessment context for a liability clause")
    print("=" * 60)

    test_liability = (
        "The aggregate liability of the Vendor under this Agreement "
        "shall not exceed the total fees paid by the Client in the "
        "twelve (12) months preceding the claim."
    )
    context = retriever.get_risk_assessment_context(
        clause_text=test_liability,
        category="Cap On Liability",
        n_similar=3,
    )
    print(context)

    # Test 3: Cross-category search (no filter)
    print("\n" + "=" * 60)
    print("Test 3: Cross-category search (no category filter)")
    print("=" * 60)

    test_ip = "All intellectual property created during the term shall belong to the Company."
    results = retriever.find_similar_clauses(
        clause_text=test_ip,
        n_results=5,
    )
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] Category: {r['category']} | Distance: {r['distance']:.4f}")
        print(f"      Text: {r['clause_text'][:150]}...")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()
