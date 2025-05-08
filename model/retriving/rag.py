# rag/rag.py
import time
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from utils.env_loader import load_env
from utils.prompt_loader import load_prompt
from utils.llm_loader import load_llm  
from langchain.prompts import PromptTemplate


def build_rag_chain(retriever, chain_type="stuff", prompt_type="default"):

    """
    - retriever: LlamaIndexRetriever 등 LangChain Retriever
    - chain_type: stuff / map_reduce / refine
    - prompt_type: default / markdown / json
    """

    start = time.time()
    llm = load_llm(temperature=0.2, top_p=0.95, max_tokens=1024) 
    print(f"⏱️ LLM load time: {time.time() - start:.2f}s")

    start = time.time()
    prompt_text = load_prompt(f"prompts/{prompt_type}_prompt.txt")  # 예: prompts/default_prompt.txt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_text,
    )
    print(f"⏱️ Prompt load + setup time: {time.time() - start:.2f}s")

    start = time.time()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type=chain_type,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,  
    )
    print(f"⏱️ RetrievalQA chain build time: {time.time() - start:.2f}s")

    return chain
