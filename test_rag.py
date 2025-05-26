import json
import os
from utils.import_loader import load_config, dynamic_import
from utils.import_loader import load_modules_from_config
from llama_index.core.schema import Document
from ragas import evaluate
from ragas.metrics import context_recall, faithfulness, answer_correctness
from ragas.llms import LangchainLLMWrapper
from sentence_transformers import SentenceTransformer
from ragas import EvaluationDataset, SingleTurnSample

def load_parsed_markdown(directory: str):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            with open(os.path.join(directory, filename), encoding="utf-8") as f:
                content = f.read()
                documents.append(Document(page_content=content, metadata={"source": filename}))
    return documents

config = load_config()
modules = load_modules_from_config()
parser_conf = config['modules']['parser']
embedding_conf = config['modules']['embeddings']

parser = dynamic_import(parser_conf['module'], parser_conf['functions'])
embedding = dynamic_import(embedding_conf['module'], embedding_conf['functions'])

load_all_pdfs = parser['load_all_pdfs']
load_retriever = modules['retriever']['load_retriever']
create_medical_chain = modules['chains']['create_medical_chain']
get_chain_response = modules['chains']['get_chain_response']
create_ollama_llm = modules['chains']['create_ollama_llm']
embed_texts = embedding['embed_texts']

# Load PDF
# documents = load_all_pdfs("pdf_data")
documents = load_parsed_markdown("data/processed")
retriever = load_retriever(k=5)
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
    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=[doc.page_content for doc in docs],
        reference=item["reference_answer"],
        response=answer
    )
    predictions.append(sample)
    # predictions.append({
    #     "question": item["question"],
    #     "answer": answer,
    #     "ground_truth": item["reference_answer"],
    #     "context": [doc.page_content for doc in docs]
    # })

evaluator_llm = LangchainLLMWrapper(llm)
dataset = EvaluationDataset(predictions)
# help(EvaluationDataset)

results = evaluate(
    dataset=dataset,
    llm=evaluator_llm,
    embeddings=SentenceTransformer("dragonkue/BGE-m3-ko"),
    metrics=[
        context_recall,
        faithfulness,
        answer_correctness
    ]
)
print(results)