from docling_.parser import load_all_pdfs
from rag.retrieval import build_retriever
from rag.rag import build_rag_chain

if __name__ == "__main__":
    documents = load_all_pdfs("pdf_data")
    retriever = build_retriever(documents)
    qa_chain = build_rag_chain(retriever)

    print("🔍 PDF 기반 질의응답 시작!")
    while True:
        query = input("질문 ▶ ")
        print("📘 답변:", qa_chain.run(query))
