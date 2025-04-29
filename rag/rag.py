# rag/rag.py
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from utils.env_loader import load_env
from utils.prompt_loader import load_prompt
from langchain.prompts import PromptTemplate


def build_rag_chain(retriever, chain_type="stuff", prompt_type="default"):

    """
    - retriever: LlamaIndexRetriever 등 LangChain Retriever
    - llm: LangChain LLM 객체 (ex: Ollama, OpenAI)
    - chain_type: stuff / map_reduce / refine
    - prompt_type: default / markdown / json
    """

    prompt_text = load_prompt(f"prompts/{prompt_type}_prompt.txt")  # 예: prompts/default_prompt.txt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_text,
    )

    base_url, model_name = load_env()
    llm = OllamaLLM(model=model_name, base_url=base_url)
    custom_prompt = get_custom_prompt(prompt_type)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type=chain_type,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,  
    )
    return chain
