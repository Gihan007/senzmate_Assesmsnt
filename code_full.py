# all the imports
import streamlit as st
import os
import weaviate
from weaviate.auth import AuthApiKey
from dotenv import load_dotenv
from io import BytesIO
from docx import Document
from PyPDF2 import PdfReader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Weaviate
from groq import Groq

# Load environment variables
load_dotenv()

# API keys
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Weaviate params
WEAVIATE_URL = "https://boho6ox7rmufpc3ezzzsqq.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "Wk0yR3BFcTFwckRIZDJTWF9tTW85U3EvdUVaTkIyZ3p2ZDVMaSsxNjFmWDNTalRUSll2TTg3WElXS0tzPV92MjAw"

# Initialize Weaviate client once
client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=AuthApiKey(WEAVIATE_API_KEY),
)

# Initialize Groq client once
groq_client = Groq(api_key=groq_api_key)


def process_input(input_type, input_data):
    documents = []

    if input_type == "Link":
        for url in input_data:
            loader = WebBaseLoader(url)
            docs = loader.load()
            documents.extend(docs)

    elif input_type == "PDF":
        pdf_reader = PdfReader(BytesIO(input_data.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        documents = [text]

    elif input_type == "Text":
        documents = [input_data]

    elif input_type == "DOCX":
        doc = Document(BytesIO(input_data.read()))
        text = "\n".join([para.text for para in doc.paragraphs])
        documents = [text]

    elif input_type == "TXT":
        text = input_data.read().decode('utf-8')
        documents = [text]

    else:
        raise ValueError("Unsupported input type")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    if input_type == "Link":
        texts = text_splitter.split_documents(documents)
        texts = [str(doc.page_content) for doc in texts]
    else:
        combined_text = "\n".join(documents)
        texts = text_splitter.split_text(combined_text)

    huggingface_embeddings = HuggingFaceEmbeddings()

    vector_db = Weaviate.from_texts(
        texts,
        embedding=huggingface_embeddings,
        client=client
    )

    return vector_db


def answer_question(vector_db, conversation_history, query):
    retriever = vector_db.as_retriever()
    relevant_docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Build chat messages for Groq (with full chat history)
    messages = [{"role": "system", "content": (
        "You are a helpful AI assistant. Answer ONLY based on the provided CONTEXT. "
        "If the answer is not in the CONTEXT, respond ONLY with: 'I don't know based on the provided documents.' "
        "Provide detailed, thorough, and clear explanations in your answers. "
        "Use multiple sentences or paragraphs as needed. Provide the FINAL answer."
    )}]

    # Add previous chat messages
    for chat in conversation_history:
        messages.append({"role": "user", "content": chat["user"]})
        messages.append({"role": "assistant", "content": chat["assistant"]})

    # Add current user message
    user_content = (
        f"Answer the following question using ONLY the CONTEXT below.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        f"Provide the FINAL answer."
    )
    messages.append({"role": "user", "content": user_content})

    # Call Groq
    completion = groq_client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=messages,
        temperature=0.5,
    )

    answer = completion.choices[0].message.content
    clean_answer = answer.replace("<think>", "").replace("</think>", "").strip()

    return clean_answer


# MAIN APP
def main():
    st.set_page_config(page_title="🧠 RAG ChatBot", page_icon="💬", layout="wide")

    # CSS Styling
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        color: #4a90e2;
        font-size: 42px;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        margin-top: 20px;
        color: gray;
    }
    .chat-bubble-user {
        background-color: #d1e9ff;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 80%;
        align-self: flex-end;
    }
    .chat-bubble-assistant {
        background-color: #f4f4f4;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 80%;
        align-self: flex-start;
    }
    </style>
    """, unsafe_allow_html=True)

    # App title
    st.markdown('<div class="title">💬 RAG ChatBot - Conversational AI</div>', unsafe_allow_html=True)

    st.sidebar.header("📂 Upload Data First")
    input_type = st.sidebar.selectbox("Select Input Type", ["Link", "PDF", "Text", "DOCX", "TXT"])

    if input_type == "Link":
        number_input = st.sidebar.number_input("Number of Links", min_value=1, max_value=20, step=1)
        input_data = []
        for i in range(number_input):
            url = st.sidebar.text_input(f"URL {i+1}")
            if url:
                input_data.append(url)

    elif input_type == "Text":
        input_data = st.text_area("Enter the text")

    elif input_type == 'PDF':
        input_data = st.file_uploader("Upload a PDF file", type=["pdf"])

    elif input_type == 'TXT':
        input_data = st.file_uploader("Upload a text file", type=['txt'])

    elif input_type == 'DOCX':
        input_data = st.file_uploader("Upload a DOCX file", type=['docx', 'doc'])

    # Vector DB creation button
    if st.sidebar.button("🚀 Process Document"):
        if input_data:
            with st.spinner("Processing input..."):
                vectorstore = process_input(input_type, input_data)
                st.session_state["vector_db"] = vectorstore
                st.session_state["chat_history"] = []  # Initialize chat history
            st.sidebar.success("✅ Vector store ready!")
        else:
            st.sidebar.warning("⚠️ Please provide valid input.")

    # Chat section
    if "vector_db" in st.session_state:

        # Display previous chat messages
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        st.write("### Chat History")
        for chat in st.session_state["chat_history"]:
            st.markdown(f'<div class="chat-bubble-user">🧑‍💻 You: {chat["user"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble-assistant">🤖 Bot: {chat["assistant"]}</div>', unsafe_allow_html=True)

        # Input new question
        query = st.text_input("💬 Ask something", placeholder="Type your question here...")
        if st.button("Submit"):
            if query.strip() != "":
                with st.spinner("Thinking... 🤔"):
                    answer = answer_question(
                        st.session_state["vector_db"],
                        st.session_state["chat_history"],
                        query
                    )
                # Save to chat history
                st.session_state["chat_history"].append({
                    "user": query,
                    "assistant": answer
                })
                # Re-render chat
                st.rerun()


    st.markdown('<div class="footer">Developed for Senzmate Interview Demo by Gihan Lakmal 🚀</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
