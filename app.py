import streamlit as st
import time
from typing import List, Dict
from utils.import_loader import load_modules_from_config
from utils.import_loader import load_config, dynamic_import
import sys

# print("APP #1")
config = load_config()
# print("APP #2")
modules = load_modules_from_config()
# print("APP #3")
# parser_conf = config['modules']['parser']
# embedding_conf = config['modules']['embeddings']
# parser = dynamic_import(parser_conf['module'], parser_conf['functions'])
# embedding = dynamic_import(embedding_conf['module'], embedding_conf['functions'])

hybrid_retriever = modules['retriever']['hybrid_retriever']
create_medical_chain = modules['chains']['create_medical_chain']
get_chain_response = modules['chains']['get_chain_response']
create_ollama_llm = modules['chains']['create_ollama_llm']
create_openai_llm = modules['chains'].get('create_openai_llm')
# print("APP #4")
# embed_texts = embedding['embed_texts']

# Set page config
st.set_page_config(
    page_title="PDF RAG Chat",
    page_icon="📚",
    layout="wide"
)
# print("APP #5")

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
    # Select LLM based on command-line argument
    if 'gpt' in sys.argv and create_openai_llm is not None:
        # print("LOAD LLM - GPT")
        llm = create_openai_llm()
        # print("GPT model loaded")
    else:
        llm = create_ollama_llm()
    # print("DEFINING RETRIEVER")
    retriever = hybrid_retriever(index_name=config['pinecone']['index_name'], model_name=config['pinecone']['model_name'])
    # print("DEFINING CHAIN")
    qa_chain = create_medical_chain(retriever=retriever, llm=llm)
    return retriever, qa_chain

# Load components
with st.spinner("Loading components..."):
    # print("APP #6")
    retriever, qa_chain = load_components()
    # print("APP #7")

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
            message_placeholder.markdown("🔍 의료 진료지침 문서를 검색 중입니다...")
            with st.spinner("관련 의료 정보를 찾고 있습니다..."):
                documents = retriever.get_relevant_documents(prompt)
                print(documents)
                response = get_chain_response(qa_chain, prompt, chat_history, documents)   
                print(response) 
            # Clear status message
            message_placeholder.empty()
            
            # Simulate streaming with progress bar
            #progress_bar = st.progress(0)
            words = response.split()
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