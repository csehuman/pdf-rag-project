from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging
from langchain_teddynote.korean import stopwords
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob
import re


# LangSmith 추적용 프로젝트 이름 설정
logging.langsmith("pinecone")

# 한글 불용어 사전 불러오기 (불용어 사전 출처: https://www.ranks.nl/stopwords/korean)
stopword = stopwords()


def parse_metadata_and_pages(text):
    lines = text.split('\n')
    pages = []
    current_page = {'page': None, 'content': [], 'titles': [], 'tables': []}
    
    for line in lines:
        # 페이지 토큰
        page_match = re.match(r'<PAGE>(\d+)', line)
        if page_match:
            # 이전 페이지 저장
            if current_page['page'] is not None:
                pages.append(current_page)
            # 새 페이지 초기화
            current_page = {'page': int(page_match.group(1)), 'content': [], 'titles': [], 'tables': []}
            # 페이지 토큰 라인에서 페이지 번호 이후 텍스트가 있으면 같이 처리 (예: <PAGE>3 ## 표11. ...)
            remainder = line[page_match.end():].strip()
            if remainder:
                # 제목/소제목 체크
                header_match = re.match(r'^(#{1,3})\s+(.*)', remainder)
                if header_match:
                    level = len(header_match.group(1))
                    title = header_match.group(2).strip()
                    current_page['titles'].append({'level': level, 'title': title})
                    if level == 2 and title.startswith('표'):
                        current_page['tables'].append(title)
                current_page['content'].append(remainder)
            continue
        
        # 제목/소제목
        header_match = re.match(r'^(#{1,3})\s+(.*)', line)
        if header_match:
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            current_page['titles'].append({'level': level, 'title': title})
            if level == 2 and title.startswith('표'):
                current_page['tables'].append(title)
        
        current_page['content'].append(line)
    
    # 마지막 페이지 저장
    if current_page['page'] is not None:
        pages.append(current_page)
    
    return pages


# ------------------------------------
# **데이터 전처리**
# ------------------------------------
# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

split_docs = []

# 텍스트 파일을 load -> List[Document] 형태로 변환
files = sorted(glob.glob("./pinecone_test/data/*.md"))

# for file in files:
#     loader = TextLoader(file, encoding="utf-8")
#     documents = loader.load()
#     split_docs.extend(text_splitter.split_documents(documents))

for file in files:
    loader = TextLoader(file, encoding="utf-8")
    documents = loader.load()  # List[Document], 보통 문서 1개
    
    for doc in documents:
        pages = parse_metadata_and_pages(doc.page_content)
        for page in pages:
            page_text = "\n".join(page['content'])
            # 한 페이지 텍스트를 쪼개기
            chunks = text_splitter.split_text(page_text)
            for chunk in chunks:
                # chunk 별로 Document 생성 + 메타데이터 추가
                split_docs.append({
                    'page_content': chunk,
                    'metadata': {
                        'page': page['page'],
                        'titles': page['titles'],
                        'tables': page['tables'],
                        'source_file': file
                    }
                })
print(split_docs[1]['page_content'])
print(split_docs[1]['metadata'])

