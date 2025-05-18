import time
from utils.import_loader import load_config, dynamic_import
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE
DATA_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "documents"

# 설정 로딩
config = load_config()

# 모듈 임포트
parser_conf = config['modules']['parser']
chunking_conf = config['modules']['chunking']
embedding_conf = config['modules']['embeddings']
vector_store_conf = config['modules']['vector_store']

# 동적 임포트
parser = dynamic_import(parser_conf['module'], parser_conf['functions'])
chunking = dynamic_import(chunking_conf['module'], chunking_conf['functions'])
embedding = dynamic_import(embedding_conf['module'], embedding_conf['functions'])
vector_store = dynamic_import(vector_store_conf['module'], vector_store_conf['functions'])

# 함수 매핑
load_all_pdfs = parser['load_all_pdfs']
chunk_documents = chunking['chunk_documents']
embed_texts = embedding['embed_texts']
save_faiss_index = vector_store['save_faiss_index']


def build_and_save_index():
    start = time.time()
    
    # PDF 로딩
    docs_texts = load_all_pdfs(DATA_DOCUMENTS_PATH)

    # 청킹
    chunks = chunk_documents(docs_texts)
    print(f"📏 청크 수: {len(chunks)}")
    
    # 임베딩 생성
    embeddings = embed_texts(chunks, "dragonkue/BGE-m3-ko")
    print("⚡ 임베딩 완료")
    
    # FAISS 인덱스 저장
    save_faiss_index(chunks, embeddings)
    print("💾 FAISS 저장 완료")
    
    print(f"⏳ 전체 소요 시간: {time.time() - start:.2f}s")

if __name__ == "__main__":
    # print(DATA_DOCUMENTS_PATH)
    build_and_save_index()