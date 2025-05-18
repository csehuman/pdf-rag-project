import os
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
FAISS_STORE_PATH = PROJECT_ROOT / "data" / "faiss_store"

class PrecomputedEmbeddings(Embeddings):
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.index = 0

    def embed_documents(self, texts):
        batch = self.embeddings[self.index: self.index + len(texts)]
        self.index += len(texts)
        return batch.tolist()

    def embed_query(self, text):
        raise NotImplementedError("This embedding is precomputed.")

def save_faiss_index(chunks, embeddings):
    """
    청크와 임베딩을 FAISS 인덱스로 저장합니다.
    """
    embedding_model = PrecomputedEmbeddings(embeddings)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    vectorstore.save_local(FAISS_STORE_PATH)

