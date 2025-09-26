# Chat-with-your-Data-using-RAG
This project is an interactive chatbot that allows users to upload any PDF document and ask questions about its content.
It uses:
- LangChain for text splitting, embeddings, and retrieval
- ChromaDB for storing and retrieving vectorized chunks of the PDF
- Ollama (with models like llama3.2) as the LLM for generating responses
- Streamlit to provide a simple and user-friendly chat interface

✨ Features

- Upload any PDF and chat with it instantly

- Retrieves the most relevant document chunks for accurate answers

- Interactive chat interface powered by Streamlit

- Uses HuggingFace sentence-transformers for embeddings

- Locally runs with Ollama – no external API costs

⚙️ Tech Stack

- Python 3.10+

- LangChain

- ChromaDB

- Ollama (local LLM runner)

- Streamlit

🚀 How to Run

Clone the repository:

git clone https://github.com/Wasi-Abbas/Chat-with-your-Data-using-RAG
cd pdf-assistant-ollama


📌 Install dependencies:

- pip install -r requirements.txt

- Make sure Ollama is installed and running on your system: ollama run llama3.2
  
- Run the Streamlit app:

- streamlit run app.py

Open the local URL (default: http://localhost:8501) and upload your PDF to start chatting.

📌 Example Use Cases

- Summarizing research papers

- Extracting insights from technical manuals

- Conversational Q&A on training materials

- Learning from textbooks interactively
