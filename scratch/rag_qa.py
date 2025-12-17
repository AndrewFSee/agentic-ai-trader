# rag_qa.py

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import HumanMessage

from dotenv import load_dotenv
load_dotenv()

CHROMA_DIR = "db/books"
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-5.1"   # or gpt-4.1, etc.


def load_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectordb


def rag_answer(question: str, k: int = 5) -> str:
    vectordb = load_vectorstore()
    llm = ChatOpenAI(model=CHAT_MODEL)

    # 1) Retrieve relevant chunks
    docs = vectordb.similarity_search(question, k=k)

    context = "\n\n---\n\n".join(
        [
            f"[Book: {d.metadata.get('book_name', d.metadata.get('source'))}, "
            f"page {d.metadata.get('page', '?')}] {d.page_content}"
            for d in docs
        ]
    )

    system_prompt = (
        "You are a trading assistant that ONLY uses the provided book excerpts. "
        "Base your answer strictly on them. "
        "If the excerpts are not sufficient, say you are unsure.\n\n"
        "When possible, refer to the book name and page number when explaining."
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Here are relevant excerpts from trading books:\n\n{context}\n\n"
        "Now answer the question using ONLY the information from these excerpts."
    )

    messages = [
        # If you want, you can wrap system prompt in a SystemMessage;
        # here we just prepend it.
        HumanMessage(content=system_prompt + "\n\n" + user_prompt)
    ]
    response = llm(messages)
    return response.content


if __name__ == "__main__":
    while True:
        q = input("\n❓ Ask a question about trading (or 'q' to quit): ")
        if q.lower().strip() in {"q", "quit", "exit"}:
            break
        ans = rag_answer(q)
        print("\n🤖 Answer:\n")
        print(ans)
