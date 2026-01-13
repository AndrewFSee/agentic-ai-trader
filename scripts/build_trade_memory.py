#!/usr/bin/env python
"""
Build Trade Memory Index

Creates a LlamaIndex VectorStoreIndex from the trade log (JSONL).
Each trade becomes a Document with text describing the trade context,
outcome, and agent rationale.

Usage:
    python scripts/build_trade_memory.py
    python scripts/build_trade_memory.py --log-path live_testing/trade_log.jsonl
    python scripts/build_trade_memory.py --db-dir db/trades

The index can then be loaded by the Reflexion agent to retrieve
relevant past trades when making new decisions.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.openai import OpenAIEmbedding

from live_testing.trade_logging import (
    TradeRecord,
    load_trade_records_list,
    DEFAULT_LOG_PATH,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DB_DIR = Path(__file__).parent.parent / "db" / "trades"
EMBED_MODEL = "text-embedding-3-large"


def trade_to_document(record: TradeRecord) -> Document:
    """
    Convert a TradeRecord to a LlamaIndex Document.
    
    The document text is natural language describing the trade,
    suitable for embedding and similarity search.
    """
    text = record.to_document_text()
    metadata = record.get_metadata()
    
    return Document(
        text=text,
        metadata=metadata,
        doc_id=record.trade_id,
    )


def build_trade_memory_index(
    log_path: Path = DEFAULT_LOG_PATH,
    db_dir: Path = DEFAULT_DB_DIR,
) -> VectorStoreIndex:
    """
    Build or rebuild the trade memory index from trade logs.
    
    Args:
        log_path: Path to the JSONL trade log
        db_dir: Directory to persist the index
        
    Returns:
        The created VectorStoreIndex
    """
    logger.info("Loading trade records from %s", log_path)
    
    # Load all trade records
    records = load_trade_records_list(log_path)
    logger.info("Loaded %d trade records", len(records))
    
    if not records:
        logger.warning("No trade records found. Index will be empty.")
        # Create empty index
        documents = []
    else:
        # Convert to documents
        documents = [trade_to_document(r) for r in records]
        logger.info("Created %d documents for indexing", len(documents))
        
        # Log some stats
        winners = sum(1 for r in records if r.pnl > 0)
        losers = len(records) - winners
        total_pnl = sum(r.pnl for r in records)
        logger.info(
            "Stats: %d winners, %d losers, total P&L: $%.2f",
            winners, losers, total_pnl
        )
    
    # Configure embedding model
    logger.info("Configuring embedding model: %s", EMBED_MODEL)
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)
    
    # Create the index
    logger.info("Building vector index...")
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )
    
    # Persist to disk
    logger.info("Persisting index to %s", db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(db_dir))
    
    logger.info("Trade memory index built successfully!")
    logger.info("  - Documents: %d", len(documents))
    logger.info("  - Location: %s", db_dir)
    
    return index


def load_trade_memory_index(
    db_dir: Path = DEFAULT_DB_DIR,
) -> VectorStoreIndex:
    """
    Load an existing trade memory index from disk.
    
    Args:
        db_dir: Directory containing the persisted index
        
    Returns:
        The loaded VectorStoreIndex
        
    Raises:
        FileNotFoundError: If the index doesn't exist
    """
    if not db_dir.exists():
        raise FileNotFoundError(f"Trade memory index not found at {db_dir}")
    
    logger.info("Loading trade memory index from %s", db_dir)
    
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)
    storage_context = StorageContext.from_defaults(persist_dir=str(db_dir))
    index = VectorStoreIndex.from_storage_context(storage_context)
    
    logger.info("Loaded trade memory index")
    return index


def query_trade_memory(
    query: str,
    k: int = 5,
    db_dir: Path = DEFAULT_DB_DIR,
) -> List[dict]:
    """
    Query the trade memory for relevant past trades.
    
    Args:
        query: Natural language query
        k: Number of results to return
        db_dir: Directory containing the index
        
    Returns:
        List of dicts with 'text', 'score', and 'metadata' keys
    """
    index = load_trade_memory_index(db_dir)
    retriever = index.as_retriever(similarity_top_k=k)
    
    results = retriever.retrieve(query)
    
    return [
        {
            "text": r.node.get_content(),
            "score": r.score,
            "metadata": dict(r.node.metadata) if r.node.metadata else {},
        }
        for r in results
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Build trade memory index from trade logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"Path to trade log JSONL (default: {DEFAULT_LOG_PATH})"
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=DEFAULT_DB_DIR,
        help=f"Directory to store index (default: {DEFAULT_DB_DIR})"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Optional: Test query to run after building"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Build the index
    index = build_trade_memory_index(
        log_path=args.log_path,
        db_dir=args.db_dir,
    )
    
    # Optional test query
    if args.query:
        print(f"\n{'='*60}")
        print(f"Test Query: {args.query}")
        print(f"{'='*60}\n")
        
        results = query_trade_memory(
            query=args.query,
            k=3,
            db_dir=args.db_dir,
        )
        
        for i, r in enumerate(results, 1):
            print(f"--- Result {i} (score: {r['score']:.4f}) ---")
            print(r['text'][:500])
            print()


if __name__ == "__main__":
    main()
