import streamlit as st
import time
from typing import List, Dict
from utils.import_loader import load_modules_from_config
import asyncio
import re
import inspect


modules = load_modules_from_config()

load_retriever = modules['retriever']['load_retriever']
create_classifier_chain = modules['chains']['create_classifier_chain']
create_medical_chain = modules['chains']['create_medical_chain']
create_general_chain = modules['chains']['create_general_chain']
get_chain_response = modules['chains']['get_chain_response']
create_ollama_llm = modules['chains']['create_ollama_llm']

# Set page config
st.set_page_config(
    page_title="PDF RAG Chat",
    page_icon="📚",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []      

def format_chat_history(messages: List[Dict]) -> str:
    """Format chat history for the LLM prompt."""
    formatted = []
    for msg in messages:
        role = "사용자" if msg["role"] == "user" else "시스템"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted[-6:])  # Keep last 3 QA pairs

# Initialize components
@st.cache_resource
def load_components():
    llm = create_ollama_llm()
    retriever = load_retriever(k=5)
    classifier_chain = create_classifier_chain(llm)
    medical_chain = create_medical_chain(llm, retriever)
    general_chain = create_general_chain(llm)
    return retriever, classifier_chain, medical_chain, general_chain

# Load components
with st.spinner("Loading components..."):
    retriever, classifier_chain, medical_chain, general_chain = load_components()

# Title and description
st.title("📚 PDF RAG Chat")
st.markdown("""
Ask questions about your PDF documents. The system will retrieve relevant information and provide answers.
""")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Format chat history
            chat_history = format_chat_history(st.session_state.messages[:-1])
            
            # Step 1: Classify the question
            message_placeholder.markdown("🤔 질문을 분석하고 있습니다...")
            
            classification = get_chain_response(classifier_chain, prompt, chat_history)
            category = re.search(r"CATEGORY:\s*(\d+)", classification["text"]).group(1)
            print('classification', classification)
            print(category)
            
            # Step 2: Process based on classification
            if category == "1":  # Medical question
                message_placeholder.markdown("🔍 의료 진료지침 문서를 검색 중입니다...")
                with st.spinner("관련 의료 정보를 찾고 있습니다..."):
                    print('documents11')
                    print(type(retriever))
                    print(inspect.getsource(retriever.invoke))
                    documents = retriever.invoke(input="폐암의 방사선 치료원칙은 무엇인가요?")
                    print('documents')
                    
                    response = get_chain_response(medical_chain, prompt, chat_history, documents)
                    print('response')
            else:  # General or conversation-related question
                message_placeholder.markdown("💭 답변을 생성하고 있습니다...")
                response = get_chain_response(general_chain, prompt, chat_history)
            
            # Clear status message
            message_placeholder.empty()
            
            # Simulate streaming with progress bar
            #progress_bar = st.progress(0)
            
            print("[DEBUG] Response:", response)
            words=''
            if isinstance(response, str):
                # 🔍 문자열 처리
                print("[DEBUG] Result Type: str")
                words = response.split()
            elif isinstance(response, dict):
                # 🔍 딕셔너리 처리
                print("[DEBUG] Result Type: dict, Keys:", list(response.keys()))

                # "answer" 키가 있다면 우선 반환
                if "text" in response:
                    words = response["text"].split()
                else:
                    # 없을 경우 첫 번째 키를 선택
                    first_key = next(iter(response))
                    words = response[first_key].split()

            for i, chunk in enumerate(words):
                full_response += chunk + " "
                progress = (i + 1) / len(words)
                #progress_bar.progress(progress)
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            
            # Remove progress bar and show final response
            #progress_bar.empty()
            message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_message = "⚠️ 죄송합니다. 오류가 발생했습니다. 다시 시도해주세요."
            message_placeholder.markdown(error_message)
            st.error(f"Error: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": error_message}) 