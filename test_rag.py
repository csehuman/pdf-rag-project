import json
import os
import re
from utils.import_loader import load_config, dynamic_import
from utils.import_loader import load_modules_from_config
from llama_index.core.schema import Document
from ragas import evaluate
from ragas.metrics import context_recall, faithfulness, answer_correctness
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset, SingleTurnSample
from utils.env_loader import load_env

def load_parsed_markdown(directory: str):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            with open(os.path.join(directory, filename), encoding="utf-8") as f:
                content = f.read()
                documents.append(Document(page_content=content, metadata={"source": filename}))
    return documents


def clean_context(text):
    # HTML 주석 제거
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # Markdown 헤더 제거
    text = re.sub(r'^#+\s?.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'Page\s*\d+', '', text)
    
    # 공백 및 중복 구두점 정리
    text = re.sub(r'[.,]{2,}', '.', text)
    text = re.sub(r'[ ]{2,}', ' ', text)

    # 쉼표 앞의 공백 제거 → "단어 ,단어" -> "단어, 단어"
    text = re.sub(r'\s+,', ',', text)

    # 쉼표가 여러 개 반복되면 하나로 축소
    text = re.sub(r',+', ',', text)

    # 중복된 공백 정리 (공백이 2개 이상 → 1개)
    text = re.sub(r'\s{2,}', ' ', text)

    # 중복된 마침표, 느낌표, 물음표도 정리
    text = re.sub(r'([.!?]){2,}', r'\1', text)

    text = text.replace('\n', ' ')
    
    return text.strip()

config = load_config()
modules = load_modules_from_config()
parser_conf = config['modules']['parser']
embedding_conf = config['modules']['embeddings']
parser = dynamic_import(parser_conf['module'], parser_conf['functions'])
embedding = dynamic_import(embedding_conf['module'], embedding_conf['functions'])

load_all_pdfs = parser['load_all_pdfs']
load_retriever = modules['retriever']['load_retriever']
load_retriever_2 = modules['retriever']['load_retrieval_2']
create_medical_chain = modules['chains']['create_medical_chain']
get_chain_response = modules['chains']['get_chain_response']
create_ollama_llm = modules['chains']['create_ollama_llm']
embed_texts = embedding['embed_texts']

# openai로 testing llm을 설정함.
load_env()

# documents = load_all_pdfs("pdf_data")
documents = load_parsed_markdown("data/processed")
retriever = load_retriever()
llm = create_ollama_llm()
qa_chain = create_medical_chain(retriever=retriever, llm=llm)

# Load QA pairs
with open("tests/pdf_qa_dataset.json", encoding="utf-8") as f:
    dataset = json.load(f)

predictions = []
for item in dataset:
    question = item["question"]
    prompt = question
    chat_history = [] 
    docs = retriever.get_relevant_documents(question)
    answer = get_chain_response(qa_chain, prompt, chat_history, docs)
    print(docs)
    
    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=[doc.page_content for doc in docs],
        reference=item["reference_answer"],
        response=answer
    )
    predictions.append(sample)


dataset = EvaluationDataset(samples=predictions)
score = evaluate(dataset, metrics = [faithfulness, context_recall, answer_correctness])
print(score)

