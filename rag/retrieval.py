from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama as LI_Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from langchain_community.retrievers.llama_index import LlamaIndexRetriever
from .embeddings import get_korean_embedding
from utils.env_loader import load_env

def build_retriever(doc_texts: list[str], chunk_size=512, top_k=5):
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=50)
    docs = [Document(text=t) for t in doc_texts]
    nodes = splitter.get_nodes_from_documents(docs)

    embed_model = get_korean_embedding()
    index = VectorStoreIndex(nodes, embed_model=embed_model)

    # Use your Ollama LLM in Llama-Index:
    base_url, model_name = load_env()
    Settings.llm = LI_Ollama(model=model_name, base_url=base_url)

    # build a query engine (now using your phi4 model)
    query_engine = index.as_query_engine(similarity_top_k=top_k)

    # wrap for LangChain
    return LlamaIndexRetriever(index=query_engine)
