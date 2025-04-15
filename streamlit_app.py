import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.ollama import Ollama
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.readers.file import PDFReader

# --- Config ---
DOCS_DIR = "docs"
os.makedirs(DOCS_DIR, exist_ok=True)

st.set_page_config(page_title="📄 Chat with your PDF", layout="wide")
st.title("📄💬 Chat with your PDF (Local RAG using Ollama + LlamaIndex)")

# --- Session State ---
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: Upload + Model ---
with st.sidebar:
    st.header("📁 PDF & Model Setup")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    model_choice = st.selectbox("Choose Ollama Model", ["mistral", "llama2", "llama3", "gemma"])

    if st.button("📚 Load & Index PDF"):
        if uploaded_file:
            # Save file to disk
            file_path = os.path.join(DOCS_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            # Load & index
            with st.spinner("Reading and indexing PDF..."):
                reader = SimpleDirectoryReader(input_dir=DOCS_DIR, file_extractor={".pdf": PDFReader()})
                documents = reader.load_data()

                llm = Ollama(model=model_choice)
                service_context = ServiceContext.from_defaults(llm=llm)
                index = VectorStoreIndex.from_documents(documents, service_context=service_context)
                chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

                st.session_state.chat_engine = chat_engine
                st.session_state.chat_history = []

            st.success("✅ PDF indexed! Start chatting below.")
        else:
            st.warning("Please upload a PDF first.")

# --- Main Chat Interface ---
if st.session_state.chat_engine:
    st.subheader("💬 Ask a question about your PDF")
    user_input = st.chat_input("Type your question here...")

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    if user_input:
        st.chat_message("user").markdown(user_input)
        response = st.session_state.chat_engine.chat(user_input)
        st.chat_message("assistant").markdown(response.response)

        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", response.response))
else:
    st.info("👈 Upload a PDF and choose a model to start chatting.")
