from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import StuffDocumentsChain
from typing import Optional
from utils.env_loader import load_env


def create_ollama_llm():
    """Create Ollama LLM instance using environment settings."""
    base_url, model_name = load_env()
    return Ollama(
        model=model_name,
        base_url=base_url,
        temperature=0
    )

def load_prompt_template(prompt_file: str) -> str:
    """Load prompt template from file."""
    with open(f"data/prompts/{prompt_file}.txt", "r", encoding="utf-8") as f:
        return f.read()

def create_classifier_chain(llm) -> LLMChain:
    """Create a chain for classifying questions."""
    prompt = PromptTemplate(
        template=load_prompt_template("classifier_prompt"),
        input_variables=["question", "chat_history"]
    )
    return LLMChain(llm=llm, prompt=prompt)

def create_medical_chain(llm, retriever) -> StuffDocumentsChain:
    """Create a chain for medical questions with RAG."""
    prompt = PromptTemplate(
        template=load_prompt_template("medical_prompt"),
        input_variables=["context", "question", "chat_history"]
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    return StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )

def create_general_chain(llm) -> LLMChain:
    """Create a chain for general questions."""
    prompt = PromptTemplate(
        template=load_prompt_template("general_prompt"),
        input_variables=["question", "chat_history"]
    )
    return LLMChain(llm=llm, prompt=prompt)

def get_chain_response(chain, query: str, chat_history: str, documents: Optional[list] = None) -> str:
    """Get response from appropriate chain with proper inputs."""
    print('get_chain_response')
    if isinstance(chain, StuffDocumentsChain) and documents:
        print('if')
        return chain.invoke(input={
            "question": query,
            "input_documents": documents,
            "chat_history": chat_history
        })
    else:
        print('else')
        return chain.invoke(input={
            "question": query,
            "chat_history": chat_history
        }) 


if __name__ == "__main__":
    llm = create_ollama_llm()
    template = load_prompt_template("classifier_prompt")
    from utils.import_loader import load_modules_from_config
    modules = load_modules_from_config()
    load_retriever = modules['retriever']['load_retriever']
    retriever = load_retriever(k=5)
    classifier_chain = create_classifier_chain(llm)
    medical_chain = create_medical_chain(llm, retriever)
    general_chain = create_general_chain(llm)