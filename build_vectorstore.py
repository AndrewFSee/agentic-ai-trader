# build_vectorstore.py

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv

load_dotenv()

# ----- CONFIG -----
BOOKS_DIR = Path("data/books")
CHROMA_DIR = "db/books"
EMBED_MODEL = "text-embedding-3-large"  # good quality; you can switch later
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


def clean_text(s: str) -> str:
    """
    Remove invalid Unicode surrogate code points (e.g. '\\ud835') and other
    non-UTF8-encodable oddities that can crash Chroma upserts.
    """
    if s is None:
        return ""
    # 'surrogatepass' allows encoding even if surrogates exist, then 'ignore'
    # drops any characters that cannot be decoded back cleanly.
    return s.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")


def load_pdfs(books_dir: Path):
    docs = []
    for pdf_path in books_dir.glob("*.pdf"):
        print(f"📚 Loading {pdf_path.name} ...")
        loader = PyPDFLoader(str(pdf_path))

        # Each page is a Document with metadata like {"source": "...", "page": n}
        pdf_docs = loader.load()

        # Tag which book it came from + sanitize text early (best for stable chunking)
        for d in pdf_docs:
            d.metadata["book_name"] = pdf_path.stem
            d.page_content = clean_text(d.page_content)

        docs.extend(pdf_docs)

    print(f"✅ Loaded {len(docs)} pages total.")
    return docs


def split_docs(docs):
    print("✂️  Splitting into chunks ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks.")
    return chunks


def build_vectorstore(chunks):
    print("🧠 Building vector store (batched) ...")
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    # Initialize (or reuse) the Chroma collection
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    batch_size = 128  # keep this conservative to avoid hitting token limits

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_metas = metadatas[i : i + batch_size]

        # Log any problematic strings (should be rare after early cleaning)
        for j, t in enumerate(batch_texts):
            try:
                t.encode("utf-8")
            except UnicodeEncodeError as e:
                m = batch_metas[j] if j < len(batch_metas) else {}
                print(
                    f"⚠️  Bad UTF-8 in book={m.get('book_name')} "
                    f"page={m.get('page')} source={m.get('source')} "
                    f"batch={i // batch_size + 1} item={j}: {e}"
                )

        # Safety-net sanitize again (in case anything slipped through)
        batch_texts = [clean_text(t) for t in batch_texts]

        print(f"➕ Adding batch {i // batch_size + 1} ({len(batch_texts)} chunks)...")
        vectordb.add_texts(batch_texts, metadatas=batch_metas)

    vectordb.persist()
    print(f"💾 Vector store saved to: {CHROMA_DIR}")
    return vectordb


def main():
    if not BOOKS_DIR.exists():
        raise FileNotFoundError(f"Books directory not found: {BOOKS_DIR}")

    docs = load_pdfs(BOOKS_DIR)
    chunks = split_docs(docs)
    build_vectorstore(chunks)
    print("🎉 Done! Your trading-book vector store is ready.")


if __name__ == "__main__":
    main()
