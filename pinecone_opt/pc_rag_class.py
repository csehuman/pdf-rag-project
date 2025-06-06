# ---- 1. 필요한 모든 import 구문 ----

import os
from typing import List, Any, Dict
from pydantic import PrivateAttr
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---- LangChain 및 기타 패키지 import ----

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

def merge_documents(*doc_lists: List[Document], top_k: int = 10) -> List[Document]:
    """여러 문서 리스트를 병합하고 중복 제거 (내용 기반 hash)"""
    seen = set()
    merged = []
    for doc_list in doc_lists:
        for doc in doc_list:
            doc_hash = hash(doc.page_content.strip())
            if doc_hash not in seen:
                seen.add(doc_hash)
                merged.append(doc)
    return merged[:top_k]

def dot_sparse_vectors(vec1: dict, vec2: dict) -> float:
    """Pinecone 스타일 sparse 벡터 (indices + values) 간 dot product"""
    i1, v1 = vec1["indices"], vec1["values"]
    i2, v2 = vec2["indices"], vec2["values"]
    
    idx_val_1 = dict(zip(i1, v1))
    idx_val_2 = dict(zip(i2, v2))

    # 공통 인덱스에 대해서만 곱셈
    return sum(idx_val_1[k] * idx_val_2[k] for k in set(idx_val_1) & set(idx_val_2))


class BM25LangChainWrapper(BaseRetriever):
    name: str = "bm25_retriever"
    description: str = "BM25 기반 한국어 sparse 검색기"
    tags: List[str] = ["bm25", "sparse", "custom"]

    # ✅ 내부 사용 속성 (Pydantic으로 선언 필요)
    _bm25: Any = PrivateAttr()
    _contents: List[str] = PrivateAttr()
    _metadatas: List[Dict[str, Any]] = PrivateAttr()
    _doc_vectors: Any = PrivateAttr()

    def __init__(self, bm25_encoder, contents: List[str], metadatas: List[Dict[str, Any]]):
        super().__init__(
            name="bm25_retriever",
            description="BM25 기반 한국어 sparse 검색기",
            tags=["bm25", "sparse", "custom"]
        )

        # ✅ 비공개 속성으로 설정
        self._bm25 = bm25_encoder
        self._contents = contents
        self._metadatas = metadatas
        self._doc_vectors = bm25_encoder.encode_documents(contents)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        query_vector = self._bm25.encode_queries([query])[0]
        scores = [dot_sparse_vectors(doc_vec, query_vector) for doc_vec in self._doc_vectors]
        top_k_indices = np.argsort(scores)[::-1][:10]

        return [
            Document(page_content=self._contents[i], metadata=self._metadatas[i])
            for i in top_k_indices
        ]

class CrossEncoderReranker:
    def __init__(self, model_name="Dongjin-kr/ko-reranker", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_n: int = 5,
        original_scores: List[float] = None,
        original_score_weight: float = 0.3,
        min_score_threshold: float = None,
        verbose: bool = False
    ) -> List[Document]:
        pairs = [(query, doc.page_content) for doc in docs]
        encodings = self.tokenizer(
            [q for q, d in pairs],
            [d for q, d in pairs],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            rerank_scores = self.model(**encodings).logits.squeeze(-1).cpu().numpy()

        if original_scores:
            # Combine reranker + retriever scores
            rerank_scores = (1 - original_score_weight) * rerank_scores + original_score_weight * np.array(original_scores)

        scored_docs = list(zip(rerank_scores, docs))
        scored_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)

        if min_score_threshold is not None:
            scored_docs = [item for item in scored_docs if item[0] >= min_score_threshold]

        if verbose:
            print("\n📊 Reranking 결과:")
            for i, (score, doc) in enumerate(scored_docs[:top_n]):
                print(f"[{i+1}] 점수: {score:.4f} | 제목: {doc.metadata.get('heading2', '')}")

        return [doc for score, doc in scored_docs[:top_n]]

class MultiRetrieverRAGChain:
    def __init__(
        self,
        bm25_retriever,
        dense_retriever,
        reranker,
        llm_chain,  # StuffDocumentsChain
        retriever_top_k=10,
        rerank_top_n=5,
        verbose=True
    ):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.reranker = reranker
        self.llm_chain = llm_chain
        self.k = retriever_top_k
        self.n = rerank_top_n
        self.verbose = verbose

    def run(self, question: str, chat_history: str = "") -> str:
        bm25_docs = self.bm25.invoke(question)[:self.k]
        dense_docs = self.dense.invoke(question)[:self.k]

        if self.verbose:
            print(f"📄 BM25 결과: {len(bm25_docs)}, Dense 결과: {len(dense_docs)}")

        merged = merge_documents(bm25_docs, dense_docs, top_k=self.k * 2)

        reranked = self.reranker.rerank(
            query=question,
            docs=merged,
            top_n=self.n,
            verbose=self.verbose
        )

        return self.llm_chain.run(
            question=question,
            chat_history=chat_history,
            input_documents=reranked
        )
