# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer
from utils.import_loader import load_config

config = load_config()
BATCH_SIZE = config['indexing']['batch_size']

def embed_texts(chunks, model_name="dragonkue/BGE-m3-ko"):
    """
    문서 청크 리스트를 임베딩합니다.
    
    Parameters:
        chunks (List[str]): 텍스트 청크 리스트
        model_name (str): HuggingFace 모델 이름
    
    Returns:
        List[List[float]]: 임베딩된 벡터 리스트
    """
    # embedding_model = HuggingFaceEmbedding(model_name=model_name)
    # embeddings = embedding_model._embed(chunks)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, batch_size=BATCH_SIZE, show_progress_bar=True)
    
    return embeddings
