from langchain_ollama import OllamaLLM
from utils.env_loader import load_env

def load_llm(temperature=0.2, top_p=1.0, max_tokens=1024):
    """
    Ollama LLM을 로드하고 파라미터를 세팅합니다.
    
    Args:
        temperature (float): 창의성 조절. 0.0 = 보수적, 1.0 = 창의적
        top_p (float): 확률 컷오프. 1.0 = 전체 단어 고려, 0.7 = 상위 70%만
        max_tokens (int): 생성할 최대 토큰 수
    
    Returns:
        OllamaLLM 객체
    """

    base_url, model_name = load_env()

    llm = OllamaLLM(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    
    return llm
