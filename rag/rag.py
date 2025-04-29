# rag/rag.py
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from utils.env_loader import load_env

def build_rag_chain(retriever, chain_type="stuff", prompt_type="default"):

    """
    - retriever: LlamaIndexRetriever 등 LangChain Retriever
    - llm: LangChain LLM 객체 (ex: Ollama, OpenAI)
    - chain_type: stuff / map_reduce / refine
    - prompt_type: default / markdown / json
    """

    base_url, model_name = load_env()
    llm = OllamaLLM(model=model_name, base_url=base_url)
    custom_prompt = get_custom_prompt(prompt_type)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type=chain_type,
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True,  
    )
    return chain
