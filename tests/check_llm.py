from dotenv import load_dotenv
import os
from langchain.llms import Ollama

# 환경변수 로드
load_dotenv("../.env")

base_url = os.getenv("OLLAMA_BASE_URL")
model_name = os.getenv("OLLAMA_MODEL_NAME")

llm = Ollama(model=model_name, base_url=base_url)

response = llm("안녕하세요, Ollama 연결 테스트입니다.")
print("LLM 응답:", response)