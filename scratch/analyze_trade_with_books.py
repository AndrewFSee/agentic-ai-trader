# analyze_trade_with_books.py

from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import HumanMessage

from dotenv import load_dotenv
load_dotenv()

# Keep these in sync with build_vectorstore.py
CHROMA_DIR = "db/books"
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-5.1"   # or "gpt-4.1" etc.


def load_vectorstore() -> Chroma:
    """
    Load the existing Chroma vector store of trading books.
    Assumes you've already run build_vectorstore.py.
    """
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectordb


def _format_docs(docs: List) -> str:
    """
    Turn retrieved docs into a readable context string,
    including book name and page number.
    """
    parts = []
    for d in docs:
        book_name = d.metadata.get("book_name", d.metadata.get("source", "unknown"))
        page = d.metadata.get("page", "?")
        parts.append(
            f"[Book: {book_name}, page {page}]\n{d.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)


def analyze_trade_with_books(
    trading_idea: str,
    symbol: str | None = None,
    vectordb: Chroma | None = None,
    k_main: int = 6,
    k_rules: int = 6,
) -> str:
    """
    Analyze a trading idea using ONLY content from your trading books.

    Parameters
    ----------
    trading_idea : str
        Free-form description of the trade you are considering.
        e.g. "I want to buy AAPL after a pullback in an uptrend on the daily chart."
    symbol : str | None
        Optional symbol, just for context in the prompt.
    vectordb : Chroma | None
        Optionally pass an existing Chroma instance; otherwise it will be loaded.
    k_main : int
        Number of chunks to retrieve based on the idea itself.
    k_rules : int
        Number of chunks to retrieve specifically about rules / risk / psychology.
    """
    if vectordb is None:
        vectordb = load_vectorstore()

    llm = ChatOpenAI(model=CHAT_MODEL)

    # --- 1) Retrieve content directly related to the trading idea ---
    idea_query = trading_idea if symbol is None else f"{symbol} {trading_idea}"
    idea_docs = vectordb.similarity_search(idea_query, k=k_main)

    # --- 2) Retrieve extra content focused on risk/rules/psychology ---
    rules_query = (
        "risk management position sizing stop loss max risk per trade drawdown "
        "trading rules trading plan psychology discipline emotions losing streak"
    )
    rules_docs = vectordb.similarity_search(rules_query, k=k_rules)

    idea_context = _format_docs(idea_docs)
    rules_context = _format_docs(rules_docs)

    # --- 3) Build a structured prompt ---
    symbol_str = symbol or "N/A"

    system_instructions = f"""
You are a trading coach who ONLY uses the information in the provided book excerpts.
You must ground all advice in those excerpts and clearly indicate when you are unsure.

Your job is to analyze a proposed trade idea and decide whether it is reasonable
according to the trading principles, risk management rules, and psychological guidance
from these books.

NEVER guarantee profit. Always highlight risks and uncertainty.
"""

    user_prompt = f"""
TRADING IDEA
------------
Symbol: {symbol_str}
Idea: {trading_idea}

BOOK EXCERPTS – IDEA-RELATED
-----------------------------
{idea_context}

BOOK EXCERPTS – RISK MANAGEMENT & PSYCHOLOGY
--------------------------------------------
{rules_context}

TASK
----
Using ONLY the information from the excerpts above:

1. Identify the key principles from the books that apply to this idea.
2. Analyze the quality of the edge in this trade, based on the books' concepts.
3. Give detailed risk management guidance:
   - reasonable position sizing as a % of account (give a range, not an exact number),
   - stop loss placement ideas,
   - risk/reward considerations.
4. Highlight relevant psychological pitfalls (e.g., fear of missing out, revenge trading,
   overconfidence, inability to take losses).
5. Check the idea against the books' trading rules and risk principles:
   - List which rules are followed.
   - List which rules may be violated.
6. Give a clear verdict in one of these categories:
   - "NOT RECOMMENDED based on the books' principles"
   - "NEEDS MORE INFORMATION / UNCLEAR"
   - "REASONABLE IF STRICT RULES ARE FOLLOWED"

At the end, include a short checklist the trader must confirm before taking the trade.
Do NOT invent concepts that are not supported by the excerpts.
If the excerpts are insufficient, explicitly say so.
"""

    messages = [
        HumanMessage(content=system_instructions.strip() + "\n\n" + user_prompt.strip())
    ]

    response = llm(messages)
    return response.content


if __name__ == "__main__":
    # Simple CLI loop for testing
    vectordb = load_vectorstore()
    print("📘 Trading decision helper (book-based). Ctrl+C or type 'q' to exit.")
    while True:
        idea = input("\nDescribe your trading idea (or 'q' to quit): ").strip()
        if idea.lower() in {"q", "quit", "exit"}:
            break
        symbol = input("Symbol (optional, press Enter to skip): ").strip() or None
        print("\n🔍 Analyzing with your trading books...\n")
        analysis = analyze_trade_with_books(
            trading_idea=idea,
            symbol=symbol,
            vectordb=vectordb,
        )
        print("🤖 Analysis:\n")
        print(analysis)
