"""Shared RAG retriever instance — initialised once, used by all agents."""

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

_retriever = None
_initialised = False


def get_shared_retriever():
    """Get or create the shared ClauseRetriever singleton."""
    global _retriever, _initialised

    if _initialised:
        return _retriever

    _initialised = True
    try:
        from rag.vectorstore import ClauseRetriever
        _retriever = ClauseRetriever()
    except (FileNotFoundError, ImportError) as e:
        print(f"[RAG] Vector store not available: {e}")
        _retriever = None

    return _retriever