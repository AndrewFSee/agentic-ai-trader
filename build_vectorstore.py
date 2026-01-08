# build_vectorstore.py
"""
Build a LlamaIndex vector store from trading books (PDFs).
Stores the index to disk in db/books/ for retrieval by the agent.
"""

import os
from pathlib import Path

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

# ----- CONFIG -----
BOOKS_DIR = Path("data/books")
INDEX_DIR = "db/books"
EMBED_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


def clean_text(s: str) -> str:
    """
    Remove invalid Unicode surrogate code points (e.g. '\\ud835') and other
    non-UTF8-encodable oddities that can crash during indexing.
    """
    if s is None:
        return ""
    return s.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")


def build_vectorstore():
    """Build and persist a LlamaIndex VectorStoreIndex from PDF trading books."""
    
    if not BOOKS_DIR.exists():
        raise FileNotFoundError(f"Books directory not found: {BOOKS_DIR}")

    # Configure LlamaIndex settings
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)
    Settings.llm = OpenAI(model="gpt-4o-mini")  # For any query synthesis
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP

    # Load PDF documents
    print(f"📚 Loading PDFs from {BOOKS_DIR} ...")
    reader = SimpleDirectoryReader(
        input_dir=str(BOOKS_DIR),
        required_exts=[".pdf"],
        recursive=False,
    )
    documents = reader.load_data()
    print(f"✅ Loaded {len(documents)} document pages total.")

    # Clean text content and add metadata
    # LlamaIndex Documents are immutable, so we create new ones with cleaned text
    from llama_index.core.schema import Document
    cleaned_documents = []
    for doc in documents:
        cleaned_text = clean_text(doc.get_content())
        metadata = dict(doc.metadata) if doc.metadata else {}
        # Add book_name metadata from filename
        if "file_name" in metadata:
            metadata["book_name"] = Path(metadata["file_name"]).stem
        cleaned_documents.append(Document(text=cleaned_text, metadata=metadata))
    
    documents = cleaned_documents
    print(f"✅ Cleaned {len(documents)} documents.")

    # Create node parser for chunking
    node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # Build the vector index
    print("🧠 Building vector store index ...")
    index = VectorStoreIndex.from_documents(
        documents,
        node_parser=node_parser,
        show_progress=True,
    )

    # Persist to disk
    os.makedirs(INDEX_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print(f"💾 Vector store index saved to: {INDEX_DIR}")

    return index


def load_index() -> VectorStoreIndex:
    """Load an existing index from disk."""
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)
    
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)
    return index


def main():
    build_vectorstore()
    print("🎉 Done! Your trading-book vector store is ready.")


if __name__ == "__main__":
    main()
