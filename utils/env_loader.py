from dotenv import load_dotenv
import os

def load_env(dotenv_path=".env"):
    load_dotenv(dotenv_path=dotenv_path)
    base_url = os.getenv("OLLAMA_BASE_URL")
    model_name = os.getenv("OLLAMA_MODEL_NAME")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    
    if not base_url or not model_name:
        raise ValueError(".env 파일에서 OLLAMA 설정을 불러올 수 없습니다.")
    return base_url, model_name