from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# 1) FAISS 로드
BASE_DIR = os.path.dirname(__file__)
FAISS_STORE_PATH = os.path.join(BASE_DIR, "faiss_store")
hf = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

vs = FAISS.load_local(
    FAISS_STORE_PATH,
    hf,
    allow_dangerous_deserialization=True
)

# 2) docstore에서 “3.2 폐암의 방사선 치료 원칙” 텍스트가 있는 청크 추출
found = []
for _id, doc in vs.docstore._dict.items():
    txt = getattr(doc, "page_content", None) or getattr(doc, "text", "")
    if "폐암의 방사선 치료 원칙" in txt:
        found.append(( _id, txt[:200].replace("\n", " ") + "…" ))

if not found:
    print("🛑 색인된 청크 중에 “폐암의 방사선 치료 원칙” 문구가 없습니다.")
else:
    print("✅ 해당 문구가 색인된 청크들:")
    for idx, snippet in found:
        print(f" • id={idx} → {snippet}")
