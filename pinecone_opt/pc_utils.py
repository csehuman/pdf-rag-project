
import os
import yaml
import pickle
from dotenv import load_dotenv
from typing import Optional

# ---- LangChain 및 기타 패키지 import ----
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms.ollama import Ollama
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

from pc_rag_class import BM25LangChainWrapper, CrossEncoderReranker, MultiRetrieverRAGChain


def load_config(config_path="pc_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---- 3. 프롬프트 로더 ----

def load_prompt_template(prompts_dir, filename):
    path = os.path.join(prompts_dir, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ---- 4. 세팅 함수 예시 ----

def setup_dense_retriever(config):
    api_key = os.getenv(config['pinecone']['api_key_env'])
    index_name = config['paths']['pinecone_index']
    pc = Pinecone(api_key=api_key, pool_threads=config['pinecone']['pool_threads'])
    index = pc.Index(index_name)
    embedding = HuggingFaceEmbeddings(model_name=config['indexing']['embedding_model'])
    vectorstore = PineconeVectorStore(index=index, embedding=embedding, text_key="context")
    return vectorstore.as_retriever(search_kwargs={"k": config['retriever']['dense_top_k']})

def setup_bm25_retriever(config):
    with open(config['paths']['bm25_pickle'], "rb") as f:
        data = pickle.load(f)
    sparse_encoder = data["bm25"]
    doc_tuples = data["docs"]
    contents = [doc[0] for doc in doc_tuples]
    metadatas = [doc[1] for doc in doc_tuples]
    return BM25LangChainWrapper(
        bm25_encoder=sparse_encoder,
        contents=contents,
        metadatas=metadatas
    )

def load_env(dotenv_path=".env"):
    load_dotenv(dotenv_path=dotenv_path)
    base_url = os.getenv("OLLAMA_BASE_URL")
    model_name = os.getenv("OLLAMA_MODEL_NAME")
    if not base_url or not model_name:
        raise ValueError(".env 파일에서 OLLAMA 설정을 불러올 수 없습니다.")
    return base_url, model_name

def setup_llm(config):
    base_url, model_name = load_env()
    system_prompt = load_prompt_template(
        config['paths']['prompts_dir'],
        config['llm']['system_prompt_file']
    )
    return Ollama(
        model=model_name,
        base_url=base_url,
        temperature=config['llm']['temperature'],
        top_k=config['llm']['top_k'],
        top_p=config['llm']['top_p'],
        repeat_penalty=config['llm']['repeat_penalty'],
        num_ctx=config['llm']['num_ctx'],
        system=system_prompt
    )

def setup_reranker(config):
    return CrossEncoderReranker(model_name=config['reranker']['model_name'])


def setup_chains(llm, bm25, dense, reranker, config):
    classifier_chain = create_classifier_chain(
        llm, config['paths']['prompts_dir'], config['chains']['classifier_chain']['prompt_file'])
    medical_chain = MultiRetrieverRAGChain(
        bm25_retriever=bm25,
        dense_retriever=dense,
        reranker=reranker,
        llm_chain=create_medical_chain(
            llm, retriever=None,
            prompt_dir=config['paths']['prompts_dir'],
            prompt_file=config['chains']['medical_chain']['prompt_file']
        ),
        retriever_top_k=config['retriever']['bm25_top_k'],
        rerank_top_n=config['reranker']['top_n']
    )
    general_chain = create_general_chain(
        llm, config['paths']['prompts_dir'], config['chains']['general_chain']['prompt_file'])
    return classifier_chain, medical_chain, general_chain

# ---- 5. create_*_chain 함수도 config에서 프롬프트 파일명을 받게 수정 ----

def create_classifier_chain(llm, prompts_dir, prompt_file):
    prompt = PromptTemplate(
        template=load_prompt_template(prompts_dir, prompt_file),
        input_variables=["question", "chat_history"]
    )
    return LLMChain(llm=llm, prompt=prompt)

def create_general_chain(llm, prompts_dir, prompt_file):
    prompt = PromptTemplate(
        template=load_prompt_template(prompts_dir, prompt_file),
        input_variables=["question", "chat_history"]
    )
    return LLMChain(llm=llm, prompt=prompt)

def create_medical_chain(llm, retriever, prompt_dir, prompt_file):
    prompt = PromptTemplate(
        template=load_prompt_template(prompt_dir, prompt_file),
        input_variables=["context", "question", "chat_history"]
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain
    return StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )

def get_chain_response(chain, query: str, chat_history: str, documents: Optional[list] = None) -> str:
    if isinstance(chain, StuffDocumentsChain) and documents:
        return chain.run(input_documents=documents, question=query, chat_history=chat_history)
    elif isinstance(chain, StuffDocumentsChain):
        return chain.run(question=query, chat_history=chat_history)
    else:
        return chain.run(question=query, chat_history=chat_history)
    

def classify_question(classifier_chain, question, chat_history):
    result = classifier_chain.run(question=question, chat_history=chat_history)
    return result.strip().lower()

def answer_medical_question(medical_chain, question, chat_history):
    return get_chain_response(medical_chain, question, chat_history)

def answer_general_question(general_chain, question, chat_history):
    return get_chain_response(general_chain, question, chat_history)


def run_qa_pipeline(
    question,
    chat_history,
    classifier_chain,
    medical_chain,
    general_chain
):
    print("🔎 질문:", question)
    print("📥 질문 전송 중...")
    type_result = classify_question(classifier_chain, question, chat_history)
    if "의료" in type_result or "medical" in type_result:
        print("의료")
        answer = answer_medical_question(medical_chain, question, chat_history)
    else:
        answer = answer_general_question(general_chain, question, chat_history)
    print("\n💬 최종 답변:", answer)
    return answer
