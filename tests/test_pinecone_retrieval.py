from dotenv import load_dotenv

import os
import pickle

from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from kiwi import PineconeKiwiHybridRetriever

'''
Part 4
'''

load_dotenv()
api_key = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=api_key, pool_threads=30)

index_name = "ko-md-strict-multilingual-e5-large-instruct"
index = pc.Index(index_name)

save_path = f'./sparse_encoder_{index_name}.pkl'
with open(save_path, "rb") as f:
    sparse_encoder = pickle.load(f)
print(f"[load_sparse_encoder]\nLoaded Sparse Encoder from: {save_path}")

model_name = "intfloat/multilingual-e5-large-instruct"
embedder = HuggingFaceEmbeddings(
    model_name=model_name
)

pinecone_params = {
    "index": index,
    "sparse_encoder": sparse_encoder,
    "embeddings": embedder,
    "top_k": 10,
    "alpha": 0.5,
    "pc": pc
}

pinecone_retriever = PineconeKiwiHybridRetriever(**pinecone_params)

# Test the retriever
query = '''
이 사람의 성별은 여자, 나이는 54세, 증상은 어지럼증을 호소하고 있고, 몸에 경련이 있어 그리고 이전에 치료받은 이력이 없어.

현재 혈압은 150/70mmHg, 맥박은 98회/min, 호흡은 27회/min, 체온은 37.2도, 산소포화도 84%, 혈당은 132mg/dL인 상태야.

이 사람의 전체적인 상태를 고려했을 때, 지금 어떻게 진료하는게 좋을지 지침을 알려줘.
'''

search_results = pinecone_retriever.invoke(query)

# Print results
for i, doc in enumerate(search_results, 1):
    print(f"\nResult {i}:")
    print(f"Content: {doc.page_content[:200]}...")  # Print first 200 chars
    print(f"Source: {doc.metadata.get('source', 'N/A')}")
    print(f"Score: {doc.metadata.get('score', 'N/A')}")