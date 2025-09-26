
#----------------------------------------------------------------------
# The Script for RAG Implementation
#----------------------------------------------------------------------
import os
import streamlit as st
import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Path where Chroma DB will be stored
PERSIST_DIR = "./chroma_db_minilm"

# ---------------- Safe function to extract response ----------------
def ask_ollama(prompt, model="llama3.2"):
    """
    Wrapper around ollama.chat to safely extract only the assistant's text content
    """
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    # The response is usually a dict-like object
    try:
        # Newer Ollama versions return response.message.content
        if hasattr(response, "message"):
            return response.message.content.strip()
        elif isinstance(response, dict):
            return response.get("message", {}).get("content", "").strip()
        else:
            return str(response).strip()
    except Exception as e:
        return f"[Error extracting content: {e}]"


# ---------------- Function to Build / Load VectorDB ----------------
def build_or_load_db(pdf_path, reuse=True):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if reuse and os.path.exists(PERSIST_DIR):
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
    else:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(pages)

        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=PERSIST_DIR
        )

    return vectordb.as_retriever(search_kwargs={"k": 3})


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="📄 PDF Chatbot with Ollama", layout="wide")
st.title("📄 PDF Assistant (Chat with Your PDF)")

# Upload PDF (only if retriever not already built)
if "chat_ready" not in st.session_state:
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file:
        pdf_path = f"./{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Build vector DB
        retriever = build_or_load_db(pdf_path, reuse=False)
        st.session_state["retriever"] = retriever
        st.session_state["history"] = []
        st.session_state["chat_ready"] = True

        # Force rerun for chat mode
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()


# ---------------- Chat Mode ----------------
if st.session_state.get("chat_ready", False):
    retriever = st.session_state["retriever"]

    # Greeting once
    if "greeted" not in st.session_state:
        st.chat_message("assistant").write("👋 Hi there! Your PDF has been processed.")
        st.chat_message("assistant").write("You can now start asking me questions about your document.")
        st.session_state["greeted"] = True

    # Chat input
    user_input = st.chat_input("Ask me something about the PDF...")
    if user_input:
        docs = retriever.get_relevant_documents(user_input)
        context = "\n".join([d.page_content for d in docs])

        prompt = f"""
        You are a helpful assistant. Use the context below to answer the question.
        If context is not enough, say "I don’t know based on the PDF."

        Context:
        {context}

        Question:
        {user_input}
        """

        answer = ask_ollama(prompt)

        # Save conversation
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", answer))

    # Display conversation
    for role, text in st.session_state.history:
        with st.chat_message("user" if role == "You" else "assistant"):
            st.write(text)