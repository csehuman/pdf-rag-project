# ---- 1. 필요한 모든 import 구문 ----
import os
from dotenv import load_dotenv

from pc_utils import *


# ---- 6. main 함수 ----

def main(config_path="pc_config.yaml"):
    config = load_config(config_path)
    load_dotenv()

    dense_retriever = setup_dense_retriever(config)
    bm25_retriever = setup_bm25_retriever(config)
    llm = setup_llm(config)
    reranker = setup_reranker(config)
    classifier_chain, medical_chain, general_chain = setup_chains(
        llm, bm25_retriever, dense_retriever, reranker, config
    )

    question = config['demo']['question']
    chat_history = config['demo']['chat_history']
    try:
        run_qa_pipeline(
            question=question,
            chat_history=chat_history,
            classifier_chain=classifier_chain,
            medical_chain=medical_chain,
            general_chain=general_chain
        )
    except Exception as e:
        print("❌ 오류 발생:", e)

if __name__ == "__main__":
    main()
