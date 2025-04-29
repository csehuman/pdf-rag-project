from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def get_korean_embedding():
    return HuggingFaceEmbedding(model_name="dragonkue/BGE-m3-ko")
