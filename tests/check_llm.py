from dotenv import load_dotenv
import os
import time
from langchain_community.llms import Ollama

# Load .env
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=env_path)

base_url = os.getenv("OLLAMA_BASE_URL")
model_name = os.getenv("OLLAMA_MODEL_NAME")

print(f"[env] base_url: {base_url}")
print(f"[env] model_name: {model_name}")

if not base_url or not model_name:
    raise ValueError("환경변수를 .env에서 불러올 수 없습니다.")

# Initialize LLM
llm = Ollama(model=model_name, base_url=base_url)

# Measure time
start_time = time.time()

response = llm("안녕하세요, 1+1은 무엇인가요?")

end_time = time.time()
elapsed = end_time - start_time

print("\n📦 LLM 응답:", response)
print(f"⏱️ 처리 시간: {elapsed:.2f}초")
