import json
from docling_.parser import load_all_pdfs
from rag.retrieval import build_retriever
from rag.rag import build_rag_chain
from ragas import evaluate

# Load PDF
documents = load_all_pdfs("pdf_data")
retriever = build_retriever(documents)
qa_chain = build_rag_chain(retriever)

# Load QA pairs
with open("tests/pdf_qa_dataset.json", encoding="utf-8") as f:
    dataset = json.load(f)

predictions = []
for item in dataset:
    answer = qa_chain.run(item["question"])
    predictions.append({
        "question": item["question"],
        "answer": answer,
        "ground_truth": item["reference_answer"]
    })

# Evaluate
results = evaluate(
    predictions=predictions,
    metrics=["context_recall", "faithfulness", "answer_correctness"],
)
print(results)