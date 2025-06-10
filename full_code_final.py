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
import streamlit_lottie as st_lottie
import requests
import json


# These are my Api Keys that i used to run this project . Groq and Hugging face API keys
# You can get your own API keys from the respective websites If you want.

"""GROQ_API_KEY  = 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
   HUGGINGFACE_API_KEY=  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        """
   
# Load environment variables
load_dotenv()

# API keys
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# My Waveiate URL and API Key
WEAVIATE_URL = "https://boho6ox7rmufpc3ezzzsqq.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "Wk0yR3BFcTFwckRIZDJTWF9tTW85U3EvdUVaTkIyZ3p2ZDVMaSsxNjFmWDNTalRUSll2TTg3WElXS0tzPV92MjAw"

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# This is the function that processes the input data based on the type of input provided by the user.
#User can provide input as a Link, PDF, Text, DOCX or TXT file.

def process_input(input_type, input_data):
    """Processes different input types and returns a vectorstore."""
    documents = []
    
    
    # Checking If user input is a Link
    if input_type == "Link":
        for url in input_data:
            loader = WebBaseLoader(url)
            docs = loader.load()
            documents.extend(docs)\
    
    
    # Checking If user input is a PDF
    elif input_type == "PDF":
        pdf_reader = PdfReader(BytesIO(input_data.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        documents = [text]
    
    
    # Checking If user input is a Text
    elif input_type == "Text":
        documents = [input_data]
    
    
    # Checking If user input is a DOCX file
    elif input_type == "DOCX":
        doc = Document(BytesIO(input_data.read()))
        text = "\n".join([para.text for para in doc.paragraphs])
        documents = [text]
    
    
    # Checking If user input is a TXT file
    elif input_type == "TXT":
        text = input_data.read().decode('utf-8')
        documents = [text]
        
    
    # If the input type is not supported, raise an error
    else:
        raise ValueError("Unsupported input type")
    
    # Split the documents into smaller chunks for processing
    # Using CharacterTextSplitter to split the documents into smaller chunks . in here i used chunk_size=1000 and chunk_overlap=100 , better to upload texts with more than 1000 characters
    # You can change these values based on your requirements.
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    # If the input type is a Link, split the documents accordingly
    # If the input type is not a Link, combine the documents into a single string and split it
    if input_type == "Link":
        texts = text_splitter.split_documents(documents)
        texts = [str(doc.page_content) for doc in texts]
    else:
        combined_text = "\n".join(documents)
        texts = text_splitter.split_text(combined_text)
    
    
    # here im using huggingfcae embeddings to convert the text chunks into vector embeddings
    huggingface_embeddings = HuggingFaceEmbeddings()
    
    
    # Initialize Weaviate client with the provided URL and API key
    # This is the Weaviate client that connects to the Weaviate instance , and checked if the client is ready to use.
    # If the client is not ready, raise an exception.
    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=AuthApiKey(WEAVIATE_API_KEY),
    )

    if client.is_ready():
        print("✅ Connected to Weaviate!")
    else:
        print("❌ Failed to connect to Weaviate.")
        raise Exception("Weaviate connection failed.")
    
    # Create a vector store in Weaviate using the text chunks and embeddings
    # This is the vector store that stores the text chunks and their embeddings in Weaviate
    vector_db = Weaviate.from_texts(
        texts,
        embedding=huggingface_embeddings,
        client=client
    )
    return vector_db



# This function answers a question using the retrieved context from the vector store and Groq llama model.
def answer_question(vector_db, query):
    """Answers a question using retrieved context + Groq DeepSeek."""
    retriever = vector_db.as_retriever()
    relevant_docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    groq_client = Groq(api_key=groq_api_key)
    
    # Define the system and user content for the Groq chat completion
    # The system content provides instructions to the AI model on how to respond. Simple Propmpt Engineering is used here to ensure the AI responds correctly.
    system_content = (
        "You are a helpful AI assistant. Answer ONLY based on the provided CONTEXT. "
        "If the answer is not in the CONTEXT, respond ONLY with: 'I don't know based on the provided documents.' "
        "Provide detailed, thorough, and clear explanations in your answers. "
        "Use multiple sentences or paragraphs as needed to fully answer the question. "
        "Provide the FINAL answer in a professional and complete manner."
    )
    
    # The user content contains the question and the context to be used for answering
    user_content = (
        f"Answer the following question using ONLY the CONTEXT below.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        f"Provide the FINAL answer without any extra text or tags."
    )
    
    
    # Call Groq to get the answer
    # This is the Groq chat completion that generates the answer based on the system and user content
    completion = groq_client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=[
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        
        # Set the temperature to control the randomness of the output
        #lower temperature means more deterministic output, higher temperature means more creative output
        temperature=0.5,
    )
    
    # Extract the answer from the completion response
    # The answer is cleaned by removing any <think> tags and extra spaces
    answer = completion.choices[0].message.content
    
    #when i used Deepseek model to answers , i got an issue where output had that "<think>" tag in the output, so i had to remove that tag from the output , but and the i removede deepseek model and used llama 4 maverick model
    # This is the clean answer that removes any <think> tags and extra spaces
    clean_answer = answer.replace("<think>", "").replace("</think>", "").strip()

    return clean_answer


# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="RAG Q&A App", page_icon="📚", layout="wide")

    # Custom CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }
    .title {
        color: #2d6cdf;
        font-size: 50px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
        animation: fadeIn 2s ease-in-out;
    }
    .stButton button {
        background-color: #2d6cdf;
        color: white;
        font-size: 18px;
        padding: 0.6em 1.8em;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0px 4px 10px rgba(45,108,223,0.3);
    }
    .stButton button:hover {
        background-color: #1c4fbf;
        transform: scale(1.05);
        box-shadow: 0px 6px 14px rgba(28,79,191,0.4);
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #2d6cdf;
        color: white;
        text-align: center;
        padding: 12px;
        font-size: 14px;
        font-weight: 500;
        letter-spacing: 0.5px;
        box-shadow: 0px -2px 10px rgba(0,0,0,0.1);
    }
    @keyframes fadeIn {
        0% {opacity: 0;}
        100% {opacity: 1;}
    }
    .chat-bubble {
        background-color: #f0f4ff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #d0d8f0;
        font-size: 16px;
        line-height: 1.5;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.05);
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Lottie animation
    lottie_ai = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_jrpzvtqz.json")

    # Title
    st.markdown('<div class="title">📚 Advanced RAG Q&A App 🚀</div>', unsafe_allow_html=True)
    st.write("Ask intelligent questions from your documents using state-of-the-art AI! 🤖")

    # Display Lottie animation
    if lottie_ai:
        st_lottie.st_lottie(lottie_ai, height=200, key="ai_animation")

    st.sidebar.header("🔍 Input Settings")
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

    if st.button("🚀 Proceed"):
        if input_data:
            with st.spinner("Processing your input... ⏳"):
                vectorstore = process_input(input_type, input_data)
                st.session_state["vector_db"] = vectorstore
                st.success("✅ Vector store created successfully!")
        else:
            st.warning("⚠️ Please provide valid input data.")

    if "vector_db" in st.session_state:
        query = st.text_input("💬 Ask your question here")
        if st.button("🎯 Submit"):
            with st.spinner("Thinking... 🤔"):
                answer = answer_question(st.session_state["vector_db"], query)
                st.markdown("### 📝 Answer:")
                st.markdown(f'<div class="chat-bubble">{answer}</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
    Developed by Gihan Lakmal | Demo for SenzMate Interview Assesment 🚀
    </div>
    """, unsafe_allow_html=True)


# Run the main function to start the Streamlit app
if __name__ == "__main__":
    main()
