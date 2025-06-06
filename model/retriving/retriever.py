from llama_index.core import  VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama as LI_Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from langchain_community.retrievers.llama_index import LlamaIndexRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.env_loader import load_env
from utils.import_loader import load_config, dynamic_import


import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
FAISS_STORE_PATH = PROJECT_ROOT / "data" / "faiss_store"


config = load_config()

# BASE_DIR = config['paths']['root']
embedding_conf = config['modules']['embeddings']
# EMBED_MODEL_INSTANCE = HuggingFaceEmbedding(model_name="dragonkue/BGE-m3-ko")
EMBED_MODEL_INSTANCE  = HuggingFaceEmbeddings(
    model_name="dragonkue/BGE-m3-ko",
    encode_kwargs={"normalize_embeddings": True}
)

# embed_texts = embedding['embed_texts']

def build_retriever(doc_texts: list[str], chunk_size=512, top_k=5):
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=50)
    docs = [Document(text=t) for t in doc_texts]
    nodes = splitter.get_nodes_from_documents(docs)

    # embed_model = get_korean_embedding()
    index = VectorStoreIndex(nodes, settings=Settings)

    # Use your Ollama LLM in Llama-Index:
    base_url, model_name = load_env()

    # build a query engine (now using your phi4 model)
    query_engine = index.as_query_engine(similarity_top_k=top_k)

    # wrap for LangChain
    return LlamaIndexRetriever(index=query_engine)

def load_retriever(k: int = 5):
    vs = FAISS.load_local(
        FAISS_STORE_PATH,
        EMBED_MODEL_INSTANCE,
        allow_dangerous_deserialization=True
    )
    return vs.as_retriever(search_kwargs={"k": k})

if __name__ == "__main__":
    build_retriever(['hi there', 'hello world'])
    load_retriever()