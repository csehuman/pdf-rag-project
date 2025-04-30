# main.py

import os
import time
import shutil

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.rag import build_rag_chain

# where your FAISS index lives
BASE_DIR         = os.path.dirname(__file__)
FAISS_STORE_PATH = os.path.join(BASE_DIR, "faiss_store")

# the exact same embedding you used in build_index.py
EMBED_MODEL = HuggingFaceEmbeddings(
    model_name="dragonkue/BGE-m3-ko",
    model_kwargs={"device": "mps"}, 
    encode_kwargs={"normalize_embeddings": True}
)

def inspect_faiss():
    """
    Print out how many vectors you have in FAISS,
    and show the first few text chunks.
    """
    vs = FAISS.load_local(
        FAISS_STORE_PATH,
        EMBED_MODEL,
        allow_dangerous_deserialization=True
    )
    idx = vs.index
    print("⚙️ FAISS stats:")
    print(f"   • total vectors: {idx.ntotal}")
    print(f"   • vector dimension: {idx.d}\n")

    # the docstore holds your original chunks
    # internally it's a dict: id -> Document
    docs = vs.docstore._dict  
    sample_ids = list(docs.keys())[:3]
    print("⚙️ Sample chunks:")
    for _id in sample_ids:
        doc = docs[_id]
        text = getattr(doc, "page_content", None) or getattr(doc, "text", "")
        print(f"\n--- id {_id} ---")
        print(text[:200].replace("\n", " "), "…")

def clear_faiss():
    """
    Delete your faiss_store folder so you can rebuild it fresh.
    """
    if os.path.exists(FAISS_STORE_PATH):
        shutil.rmtree(FAISS_STORE_PATH)
        print("🗑️ Cleared FAISS store at", FAISS_STORE_PATH)
    else:
        print("⚠️ No FAISS store found at", FAISS_STORE_PATH)

def load_retriever(k: int = 5):
    """
    Load the FAISS index and wrap it as a LangChain retriever.
    """
    vs = FAISS.load_local(
        FAISS_STORE_PATH,
        EMBED_MODEL,
        allow_dangerous_deserialization=True
    )
    return vs.as_retriever(search_kwargs={"k": k})

if __name__ == "__main__":

    # 1) Uncomment to inspect your FAISS store before you do anything
    # inspect_faiss()

    # 2) Uncomment if you want to wipe it out and rebuild from scratch
    # clear_faiss()

    # 3) Load retriever + RAG chain
    start = time.time()
    retriever = load_retriever(k=5)
    print(f"load_retriever: {time.time() - start:.2f}s")
    qa_chain  = build_rag_chain(retriever, "stuff", "default")

    # 4) Query + measure time
    query = input("질문 ▶ ")
    start = time.time()
    result = qa_chain({"query": query})
    elapsed = time.time() - start

    # 5) Print timing
    print(f"\n⏱️ Answer time: {elapsed:.2f} seconds\n")

    # 6) Show retrieved chunks
    # print("📂 Retrieved chunks:")
    # for i, doc in enumerate(result["source_documents"], 1):
    #     content = getattr(doc, "page_content", None) or getattr(doc, "text", "")
    #     print(f"\n--- chunk #{i} ---\n{content}")

    # 7) Finally the LLM’s answer
    print("\n📘 답변:", result["result"])
