from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def get_korean_embedding():
    return HuggingFaceEmbedding(model_name="jhgan/ko-sroberta-multitask")
