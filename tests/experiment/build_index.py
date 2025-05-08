# build_index.py

import time
import os
import torch
import torch.nn.functional as F
from docling_.parser import load_all_pdfs       # docling_/parser.py
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# ─── 설정 ───
PDF_FOLDER       = os.path.join(os.path.dirname(__file__), "pdf_data")
FAISS_STORE_PATH = os.path.join(os.path.dirname(__file__), "faiss_store")
MODEL_NAME       = "dragonkue/BGE-m3-ko"
CHUNK_SIZE       = 384
OVERLAP          = 30
BATCH_SIZE       = 64
# ────────────

class PrecomputedEmbeddings(Embeddings):
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.index = 0

    def embed_documents(self, texts):
        batch = self.embeddings[self.index : self.index + len(texts)]
        self.index += len(texts)
        return batch.tolist()

    def embed_query(self, text):
        raise NotImplementedError("This embedding is precomputed.")


def build_and_save_index():
    # 1) PDF 로드
    docs_texts = load_all_pdfs(PDF_FOLDER)        # list[str]

    # 2) 청킹
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
    docs     = [Document(text=t) for t in docs_texts]
    nodes    = splitter.get_nodes_from_documents(docs)
    chunks   = [n.text for n in nodes]

    total_chars = sum(len(chunk) for chunk in chunks)
    print(f"📏 청크 수: {len(chunks)}")
    print(f"📄 총 임베딩 문자 수: {total_chars:,}")
    print(f"🧩 평균 청크 길이: {total_chars / len(chunks):.2f}")

    # 3. 배치 임베딩
    start = time.time()
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(chunks, batch_size=BATCH_SIZE, show_progress_bar=True)
    print(f"⚡ 임베딩 완료: shape={embeddings.shape}, {time.time() - start:.2f}s")

    # 4. FAISS 저장
    start = time.time()
    embedding_model = PrecomputedEmbeddings(embeddings)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    vectorstore.save_local(FAISS_STORE_PATH)
    print(f"💾 FAISS 저장 완료: {FAISS_STORE_PATH}, {time.time() - start:.2f}s")

if __name__ == "__main__":
    build_and_save_index()
