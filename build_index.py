# build_index.py

import os
from docling_.parser import load_all_pdfs       # docling_/parser.py
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# ─── 설정 ───
PDF_FOLDER       = os.path.join(os.path.dirname(__file__), "pdf_data")
FAISS_STORE_PATH = os.path.join(os.path.dirname(__file__), "faiss_store")
CHUNK_SIZE       = 512
OVERLAP          = 50
# ────────────

def build_and_save_index():
    # 1) PDF 로드
    docs_texts = load_all_pdfs(PDF_FOLDER)        # list[str]

    # 2) 청킹
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
    docs     = [Document(text=t) for t in docs_texts]
    nodes    = splitter.get_nodes_from_documents(docs)
    chunks   = [n.text for n in nodes]

    # 3) 임베딩 모델
    hf = HuggingFaceEmbeddings(model_name="dragonkue/BGE-m3-ko")

    # 4) FAISS 인덱스 빌드 및 저장
    vectorstore = FAISS.from_texts(texts=chunks, embedding=hf)
    vectorstore.save_local(FAISS_STORE_PATH)
    print(f"✅ FAISS index built & saved to {FAISS_STORE_PATH}")

if __name__ == "__main__":
    build_and_save_index()
