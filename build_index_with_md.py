import time
import glob
import os
from utils.import_loader import load_config, dynamic_import
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE
PROCESSED_MD_PATH = PROJECT_ROOT / "data" / "processed"

# 설정 로딩
config = load_config()

# 모듈 임포트
chunking_conf = config['modules']['chunking']
embedding_conf = config['modules']['embeddings']
vector_store_conf = config['modules']['vector_store']

# 동적 임포트
chunking = dynamic_import(chunking_conf['module'], chunking_conf['functions'])
embedding = dynamic_import(embedding_conf['module'], embedding_conf['functions'])
vector_store = dynamic_import(vector_store_conf['module'], vector_store_conf['functions'])

# 함수 매핑
chunk_documents = chunking['chunk_documents']
embed_texts = embedding['embed_texts']
save_faiss_index = vector_store['save_faiss_index']

def load_all_markdowns(folder_path: str) -> list[str]:
    """Load all markdown files from the given folder."""
    md_paths = glob.glob(os.path.join(folder_path, "*.md"))
    processed_files = []

    for path in md_paths:
        start_time = time.time()
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            processed_files.append(content)
        finish_time = time.time()
        processed_files.append('\n\n\n\n')
        print(f"📄 {os.path.basename(path)} 처리 완료 (소요 시간: {finish_time - start_time:.2f}초)")
    
    return processed_files

def build_and_save_index():
    start = time.time()
    
    # Markdown 로딩
    docs_texts = load_all_markdowns(PROCESSED_MD_PATH)
    print(f"📚 총 {len(docs_texts)}개의 마크다운 파일 로드 완료")

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
    build_and_save_index() 