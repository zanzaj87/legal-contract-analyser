"""
RAG Ingestion Pipeline for Legal Contract Analyser.

Downloads the CUAD (Contract Understanding Atticus Dataset) clause classification
dataset, processes it into a clause benchmark library, and stores embeddings
in a local ChromaDB vector store.

Dataset: dvgodoy/CUAD_v1_Contract_Understanding_clause_classification
- 13,155 labelled clauses from 509 commercial contracts
- 41 clause categories (e.g. Termination, Indemnification, IP Ownership)
- Sourced from SEC EDGAR filings, licensed CC BY 4.0

Usage:
    python rag/ingest.py              # Full ingestion (downloads + embeds)
    python rag/ingest.py --stats      # Show dataset statistics only
    python rag/ingest.py --dry-run    # Process without embedding (test mode)
"""

import os
import json
import hashlib
import argparse
from pathlib import Path
from collections import Counter

from datasets import load_dataset
import chromadb
from openai import OpenAI


# ─── Configuration ───────────────────────────────────────────────────────────

CHROMA_DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "clause_benchmarks"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
BATCH_SIZE = 100  # ChromaDB batch size for upserts

# Dataset source
HF_DATASET = "dvgodoy/CUAD_v1_Contract_Understanding_clause_classification"

# The 41 CUAD clause categories with descriptions for context
# These descriptions help the risk assessor understand what each category means
CATEGORY_DESCRIPTIONS = {
    "Document Name": "The name of the contract",
    "Parties": "The contracting parties",
    "Agreement Date": "The date of the contract",
    "Effective Date": "The date when the contract becomes effective",
    "Expiration Date": "The date when the contract's initial term expires",
    "Renewal Term": "The length of automatic renewal periods after the initial term",
    "Notice Period To Terminate Renewal": "The notice period required to terminate renewal",
    "Governing Law": "The state or jurisdiction whose laws govern the contract",
    "Most Favored Nation": "Whether a party is entitled to most favored nation treatment",
    "Non-Compete": "Whether a party is restricted from competing",
    "Exclusivity": "Whether there is exclusive dealing commitment",
    "No-Solicit Of Customers": "Whether a party is restricted from soliciting customers",
    "No-Solicit Of Employees": "Whether a party is restricted from soliciting employees",
    "Non-Disparagement": "Whether a party is restricted from making disparaging statements",
    "Termination For Convenience": "Whether a party can terminate without cause",
    "Rofr/Rofo/Rofn": "Whether a party has right of first refusal, offer, or negotiation",
    "Change Of Control": "Whether a change of control triggers rights or obligations",
    "Anti-Assignment": "Whether consent is required for assignment",
    "Revenue/Profit Sharing": "Whether there is revenue or profit sharing",
    "Price Restrictions": "Whether there are price restrictions or caps",
    "Minimum Commitment": "Whether there is a minimum order or commitment",
    "Volume Restriction": "Whether there are volume restrictions",
    "Ip Ownership Assignment": "Whether intellectual property ownership is assigned",
    "Joint Ip Ownership": "Whether intellectual property is jointly owned",
    "License Grant": "Whether a license is granted",
    "Non-Transferable License": "Whether the license is non-transferable",
    "Affiliate License-Licensor": "Whether the licensor can extend the license to affiliates",
    "Affiliate License-Licensee": "Whether the licensee can extend the license to affiliates",
    "Unlimited/All-You-Can-Eat-License": "Whether the license is unlimited",
    "Irrevocable Or Perpetual License": "Whether the license is irrevocable or perpetual",
    "Source Code Escrow": "Whether source code is held in escrow",
    "Post-Termination Services": "Whether services continue after termination",
    "Audit Rights": "Whether a party has audit rights",
    "Uncapped Liability": "Whether liability is uncapped",
    "Cap On Liability": "Whether there is a cap on liability",
    "Liquidated Damages": "Whether there are pre-determined damages",
    "Warranty Duration": "The duration of any warranties",
    "Insurance": "Whether insurance is required",
    "Covenant Not To Sue": "Whether a party agrees not to sue",
    "Third Party Beneficiary": "Whether there are third-party beneficiaries",
    "Matching Notice Period For Ip License-Loss": "The notice period for IP license termination",
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_contract_type(file_name: str) -> str:
    """
    Extract the contract type from the CUAD filename.

    CUAD filenames follow the pattern:
        CompanyName_DATE_Filing_EX-NUMBER_ID_EX-NUMBER_ContractType.pdf

    Examples:
        ...EX-10.B(01)_Content License Agreement.pdf → Content License Agreement
        ...EX-10.17_Co-Branding Agreement.pdf → Co-Branding Agreement
    """
    name = Path(file_name).stem  # Remove .pdf
    # The contract type is typically the last part after the final underscore
    # that contains "Agreement" or similar
    parts = name.split("_")
    # Walk backwards to find the contract type (usually the last meaningful part)
    for part in reversed(parts):
        if any(kw in part for kw in ["Agreement", "License", "Contract", "Lease", "Amendment"]):
            return part.strip()
    # Fallback: return the last part
    return parts[-1].strip() if parts else "Unknown"


def make_clause_id(file_name: str, label: str, start_at: int, row_idx: int) -> str:
    """Generate a deterministic, unique ID for each clause."""
    raw = f"{file_name}|{label}|{start_at}|{row_idx}"
    return hashlib.md5(raw.encode()).hexdigest()


def get_embedding_client() -> OpenAI:
    """Initialise OpenAI client for embeddings."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables.\n"
            "Set it with: export OPENAI_API_KEY='sk-...'"
        )
    return OpenAI(api_key=api_key)


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using OpenAI's embedding API."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


# ─── Main Pipeline ───────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def load_cuad_dataset():
    """Download and load the CUAD clause classification dataset."""
    print("📥 Downloading CUAD clause classification dataset from HuggingFace...")
    print(f"   Source: {HF_DATASET}")

    ds = load_dataset(HF_DATASET, split="train")
    print(f"   ✅ Loaded {len(ds):,} labelled clauses from {ds.num_rows:,} rows")
    return ds


def process_clauses(ds) -> list[dict]:
    """
    Process raw CUAD records into documents ready for embedding.

    Each document contains:
      - id: deterministic unique ID
      - text: the clause text (what gets embedded)
      - metadata: category, contract type, source file, etc.
    """
    print("\n🔄 Processing clauses...")

    documents = []
    skipped = 0

    for row_idx, row in enumerate(ds):
        clause_text = row["clause"].strip()

        # Skip very short clauses (likely noise or redacted content)
        if len(clause_text) < 20:
            skipped += 1
            continue

        # Skip clauses that are mostly redacted
        if clause_text.count("*") > len(clause_text) * 0.3:
            skipped += 1
            continue

        label = row["label"]
        file_name = row["file_name"]
        contract_type = get_contract_type(file_name)
        description = CATEGORY_DESCRIPTIONS.get(label, "")

        doc = {
            "id": make_clause_id(file_name, label, row["start_at"], row_idx),
            "text": clause_text,
            "metadata": {
                "category": label,
                "category_id": row["class_id"],
                "category_description": description,
                "contract_type": contract_type,
                "source_file": file_name,
                "clause_length": len(clause_text),
                "pages": row["pages"],
            },
        }
        documents.append(doc)

    print(f"   ✅ Processed {len(documents):,} clauses ({skipped} skipped)")
    return documents


def print_statistics(documents: list[dict]):
    """Print dataset statistics."""
    print("\n📊 Dataset Statistics:")
    print(f"   Total clauses: {len(documents):,}")

    # Category distribution
    categories = Counter(d["metadata"]["category"] for d in documents)
    print(f"\n   Categories ({len(categories)}):")
    for cat, count in categories.most_common():
        desc = CATEGORY_DESCRIPTIONS.get(cat, "")
        print(f"     {cat:45s} {count:5d}  — {desc}")

    # Contract type distribution
    contract_types = Counter(d["metadata"]["contract_type"] for d in documents)
    print(f"\n   Contract Types ({len(contract_types)}):")
    for ctype, count in contract_types.most_common(15):
        print(f"     {ctype:45s} {count:5d}")
    if len(contract_types) > 15:
        print(f"     ... and {len(contract_types) - 15} more")

    # Clause length stats
    lengths = [d["metadata"]["clause_length"] for d in documents]
    print(f"\n   Clause Length:")
    print(f"     Min: {min(lengths):,} chars")
    print(f"     Max: {max(lengths):,} chars")
    print(f"     Avg: {sum(lengths) // len(lengths):,} chars")
    print(f"     Median: {sorted(lengths)[len(lengths) // 2]:,} chars")


def store_in_chromadb(documents: list[dict], dry_run: bool = False):
    """
    Embed clauses and store in ChromaDB.

    Uses OpenAI text-embedding-3-small for embeddings.
    ChromaDB persists to disk at rag/chroma_db/.
    """
    if dry_run:
        print("\n🏃 Dry run — skipping embedding and storage")
        return

    print(f"\n💾 Storing {len(documents):,} clauses in ChromaDB...")
    print(f"   Path: {CHROMA_DB_PATH}")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Embedding model: {EMBEDDING_MODEL}")

    # Initialise clients
    openai_client = get_embedding_client()
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    # Delete existing collection if it exists (fresh rebuild)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print("   🗑️  Deleted existing collection (rebuilding)")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "CUAD clause benchmarks for legal contract analysis"},
    )

    # Process in batches
    total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
    total_tokens = 0

    for batch_idx in range(0, len(documents), BATCH_SIZE):
        batch = documents[batch_idx : batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1

        # Prepare texts for embedding
        # We embed: category + description + clause text
        # This gives the embedding context about what KIND of clause it is
        texts = []
        for doc in batch:
            meta = doc["metadata"]
            embedding_text = (
                f"[{meta['category']}] {meta['category_description']}\n"
                f"Contract type: {meta['contract_type']}\n"
                f"{doc['text']}"
            )
            texts.append(embedding_text)

        # Embed
        embeddings = embed_texts(openai_client, texts)

        # Prepare for ChromaDB upsert
        ids = [doc["id"] for doc in batch]
        metadatas = [doc["metadata"] for doc in batch]
        raw_texts = [doc["text"] for doc in batch]

        # Upsert into ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=raw_texts,
        )

        print(f"   Batch {batch_num}/{total_batches}: embedded {len(batch)} clauses")

    print(f"\n   ✅ Stored {collection.count():,} clauses in ChromaDB")
    print(f"   📁 Database persisted to: {CHROMA_DB_PATH}")


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest CUAD dataset into ChromaDB for RAG-enhanced contract analysis"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show dataset statistics only (no embedding)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process clauses without embedding or storing"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Legal Contract Analyser — RAG Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Download dataset
    ds = load_cuad_dataset()

    # Step 2: Process clauses
    documents = process_clauses(ds)

    # Step 3: Show statistics
    print_statistics(documents)

    if args.stats:
        return

    # Step 4: Embed and store
    store_in_chromadb(documents, dry_run=args.dry_run)

    print("\n✅ Ingestion complete!")
    print("   You can now use the vector store in the risk assessor agent.")


if __name__ == "__main__":
    main()
