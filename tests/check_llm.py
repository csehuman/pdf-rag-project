from dotenv import load_dotenv
import os
from langchain_community.llms import Ollama  # ✅ 최신 방식 (langchain 0.3.1+)

# 현재 파일 기준으로 .env 경로 설정
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=env_path)

# 환경 변수 불러오기
base_url = os.getenv("OLLAMA_BASE_URL")
model_name = os.getenv("OLLAMA_MODEL_NAME")

# 디버깅용 출력
print(f"[env] base_url: {base_url}")
print(f"[env] model_name: {model_name}")

# 환경변수 로딩 실패 시 예외 발생
if not base_url or not model_name:
    raise ValueError("환경변수를 .env에서 제대로 불러오지 못했습니다.")

# LLM 테스트
llm = Ollama(model=model_name, base_url=base_url)
response = llm("안녕하세요, Ollama 연결 테스트입니다.")
print("LLM 응답:", response)
