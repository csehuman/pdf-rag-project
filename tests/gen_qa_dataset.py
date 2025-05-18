import os
import json
from docling_.parser import load_all_pdfs
from utils.env_loader import load_env
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# 1. Load environment + init LLM
base_url, model_name = load_env()
llm = Ollama(model=model_name, base_url=base_url)

# 2. Prompt template
prompt_template = PromptTemplate.from_template("""
다음은 진료지침 문서의 일부입니다. 이 내용을 바탕으로, 다음 조건을 만족하는 질의응답 쌍을 {num_questions}개 생성하세요:

조건:
1. 질문은 해당 문서 내용을 바탕으로 하며, 구체적일 것.
2. 각 질문에 대한 답변은 최대한 문서 내용 기반으로 추론해서 작성할 것.
3. 답변은 1~3문장 내외로 구성할 것.
4. 출력은 JSON 배열 형식으로 구성할 것. 예:
[
  {{ "question": "...", "reference_answer": "..." }},
  ...
]

문서 내용:
\"\"\"
{text}
\"\"\"
""")

# 3. Load PDFs
pdf_texts = load_all_pdfs("pdf_data")  # 경로 수정 필요
qa_list = []

# 4. Generate QA for each doc
for idx, text in enumerate(pdf_texts):
    print(f"📄 문서 {idx + 1}/{len(pdf_texts)} 처리 중...")

    # 문서가 너무 길면 자르기
    text = text[:3000]

    prompt = prompt_template.format(text=text, num_questions=3)
    response = llm.invoke(prompt)

    try:
        qa_pairs = json.loads(response)
        qa_list.extend(qa_pairs)
    except json.JSONDecodeError:
        print(f"⚠️ JSON 파싱 실패 (문서 {idx + 1}) 응답: {response[:200]}...")

# 5. Save to file
with open("tests/pdf_qa_dataset.json", "w", encoding="utf-8") as f:
    json.dump(qa_list, f, ensure_ascii=False, indent=2)

print(f"\n✅ 총 {len(qa_list)}개의 QA 쌍을 생성했습니다.")
