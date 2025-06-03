from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import HierarchicalNodeParser

from llama_index.vector_stores.pinecone import PineconeVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone.enums import Metric, VectorType, CloudProvider, AwsRegion, DeletionProtection

import os
import multiprocessing


# 기본 설정
PINECONE_API_KEY = "pcsk_2taMEH_GfNAiHps89YwWisSUKSyziHmtv18Di3e68w7RJq9j4TCbGKQM9Zch1sVqngHRvX"
INDEX_NAME = "korean-medical-rag"
DOCUMENT_DIR = "./pinecone_test/data"
PROCESSED_DIR = "./pinecone_test/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

def main():
    # Pinecone 세팅
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric=Metric.COSINE,
            spec=ServerlessSpec(
                cloud=CloudProvider.AWS,
                region=AwsRegion.US_EAST_1
            ),
            deletion_protection=DeletionProtection.DISABLED,
            vector_type=VectorType.DENSE,
            tags={
                "model": "llama-text-embed-v2",
            }
        )

    # 문서 로딩 & 청킹
        # 메타데이터는 따로 뽑아내야함 - 메타데이터 추출용 파서 필요 (일단 skip)
    documents = SimpleDirectoryReader(DOCUMENT_DIR).load_data(show_progress=True, num_workers=4)
    node_parser = HierarchicalNodeParser.from_defaults()
    # node_parser = HierarchicalNodeParser(
    #     chunk_sizes=[800, 200, 100],
    #     chunk_overlap_ratio=0.1
    # )
    nodes = node_parser.get_nodes_from_documents(documents)
    for node in nodes:
        node.metadata["text"] = node.text

    # 인덱싱
    embed_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    pinecone_index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

    print("✅ 인덱싱 완료")

    index.storage_context.persist(persist_dir=PROCESSED_DIR)

if __name__ == "__main__":
    main()