# main.py

import os
from docling_.parser import load_all_pdfs      # ← pull in your parser.py
from rag.retrieval     import build_retriever
from rag.rag           import build_rag_chain

# -----------------------------------------------------------------------------
# 1) Locate pdf_data/ directory *relative* to this script
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "pdf_data")   # no more absolute paths
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # 2) Actually load *all* your PDFs into a list[str]
    documents = load_all_pdfs(PDF_FOLDER)          # returns List[str]
    
    # 3) Build retriever & QA chain as before
    retriever = build_retriever(documents)
    qa_chain  = build_rag_chain(retriever)

    # 4) Ask user, run the chain, unpack answer + sources
    query  = input("질문 ▶ ")
    result = qa_chain({"query": query})

    # 5) Print out each chunk that was retrieved
    print("\n📂 Retrieved chunks:")
    for i, doc in enumerate(result["source_documents"], 1):
        # Llama-Index docs might live in .page_content or .text
        content = getattr(doc, "page_content", None) or getattr(doc, "text", "")
        print(f"\n--- chunk #{i} ---\n{content}")

    # 6) Finally, your LLM’s answer
    print("\n📘 답변:", result["result"])
