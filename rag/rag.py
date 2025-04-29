# rag/rag.py
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from utils.env_loader import load_env

def build_rag_chain(retriever):
    base_url, model_name = load_env()
    llm = OllamaLLM(model=model_name, base_url=base_url)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,    # ← add this
    )
