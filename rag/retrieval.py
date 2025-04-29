from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from .embeddings import get_korean_embedding

def build_retriever(doc_texts: list[str], chunk_size=512, top_k=5):
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=50)
    docs = [Document(text=t) for t in doc_texts]
    nodes = splitter.get_nodes_from_documents(docs)

    embed_model = get_korean_embedding()
    index = VectorStoreIndex(nodes, embed_model=embed_model)

    return index.as_retriever(similarity_top_k=top_k)
