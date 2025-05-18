import time
from langchain_community.llms import Ollama
from utils.env_loader import load_env

base_url, model_name = load_env()

# Initialize LLM
llm = Ollama(model=model_name, base_url=base_url)

# Measure time
start_time = time.time()
response = llm("안녕하세요, 1+1은 무엇인가요?")
end_time = time.time()


print("\n📦 LLM 응답:", response)
print(f"⏱️ 처리 시간: {end_time - start_time:.2f}초")
