# from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv
import os
import time
from pinecone import Pinecone as PC

def load_env(dotenv_path=".env"):
    load_dotenv(dotenv_path=dotenv_path)
    base_url = os.getenv("OLLAMA_BASE_URL")
    model_name = os.getenv("OLLAMA_MODEL_NAME")
    if not base_url or not model_name:
        raise ValueError(".env 파일에서 OLLAMA 설정을 불러올 수 없습니다.")
    return base_url, model_name

# LLM 설정 (GCP Ollama 연결)
def create_ollama_llm():
    """Create Ollama LLM instance using environment settings."""
    base_url, model_name = load_env()
    system_prompt = load_prompt_template("system_prompt")
    return Ollama(
        model=model_name,
        base_url=base_url,
        temperature=0,
        top_k=20,
        top_p=0.5,
        repeat_penalty=1.2,
        num_ctx=4096,
        system=system_prompt
    )

def load_prompt_template(prompt_file: str) -> str:
    """Load prompt template from file."""
    with open(f"data/prompts/{prompt_file}.txt", "r", encoding="utf-8") as f:
        return f.read()

PINECONE_API_KEY = "pcsk_2taMEH_GfNAiHps89YwWisSUKSyziHmtv18Di3e68w7RJq9j4TCbGKQM9Zch1sVqngHRvX"
INDEX_NAME = "korean-medical-rag"

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OLLAMA_BASE_URL"] = "http://34.71.247.126:8888"
def main():
    # 임베딩 & 벡터 스토어 설정
    start = time.time()
    print("✅ Pinecone 연결 시작")
    pc = PC(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    print("✅ Pinecone 연결 완료:", time.time() - start, "초")

    start = time.time()
    print("✅ Embedding 로딩 중...")
    embedding = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    print("✅ Embedding 완료:", time.time() - start, "초")
    
    start = time.time()
    print("✅ Index 로딩 중...")
    vectorstore = PineconeVectorStore(index=index, embedding=embedding, text_key="text")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    print("✅ Index retrieval 완료:", time.time() - start, "초")

    start = time.time()
    print("✅ Ollama LLM 연결 중...")
    llm = create_ollama_llm()
    print("✅ LLM 연결 완료:", time.time() - start, "초")

    # QA 체인
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    # 질문 테스트
    question = "비소세포폐암 3기 치료 권고사항은 무엇인가요?"
    print("🔎 질문:", question)
    print("📥 질문 전송 중...")
    try:
        result = qa.invoke({"query": question})
        print("💬 답변:", result["result"])
        for doc in result["source_documents"]:
            print("📄 출처:", doc.metadata.get("id_"))
            print(doc.metadata.keys())
            print("----------------")
            print(doc.metadata.values())
    except Exception as e:
        print("❌ 오류 발생:", e)

if __name__ == "__main__":
    main()