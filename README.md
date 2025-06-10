# 🧠 RAG ChatBot - Conversational AI

A **Retrieval-Augmented Generation (RAG)** chatbot built with **Streamlit** that allows users to upload or link to documents (PDF, DOCX, TXT, web links, or raw text), process them into embeddings using **HuggingFace**, store in a **Weaviate** vector database, and interact with the knowledge base via a conversational AI powered by **Groq's LLM**.

---

## ✨ Features

* 🗂️ **Multiple input types:** Links, PDFs, DOCX, TXT, Text
* 🧠 **Vectorized knowledge base:** Stores document embeddings in Weaviate
* 🔍 **Contextual question answering:** Groq LLM answers based strictly on document content
* 💬 **Conversational memory:** Chat history maintained for better context
* 🚀 **Simple UI:** Streamlit-powered friendly chat interface

---

## ⚙️ Project Flow

```mermaid
graph TD
    A[User uploads data (Link, PDF, DOCX, TXT, Text)] --> B[Process Input and Split Text]
    B --> C[Generate Embeddings using HuggingFace]
    C --> D[Store embeddings in Weaviate vector DB]
    D --> E[User asks a question]
    E --> F[Retrieve relevant documents from Weaviate]
    F --> G[Build chat prompt with retrieved context + chat history]
    G --> H[Send prompt to Groq LLM]
    H --> I[LLM generates answer]
    I --> J[Answer displayed in chat interface]
```

---

## 🚀 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

### 2️⃣ (Optional but recommended) Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure environment variables

Create a `.env` file in the project root:

```env
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
GROQ_API_KEY=your_groq_api_key_here
WEAVIATE_API_KEY=your_weaviate_api_key_here
WEAVIATE_URL=https://your-weaviate-instance-url
```

---

## 🏃‍♂️ How to Run the App

1️⃣ Run the Streamlit app:

```bash
streamlit run app.py
```

2️⃣ The app will open in your default browser at:

```
http://localhost:8501
```

---

## 📝 How to Use

1. **Select Input Type** in the left sidebar:

   * **Link:** Enter one or more URLs
   * **PDF:** Upload PDF file
   * **DOCX:** Upload DOCX or DOC file
   * **TXT:** Upload TXT file
   * **Text:** Paste text directly
2. Click **🚀 Process Document** to process and create the vector database.
3. Once processed, type your question in the chat box.
4. The bot will respond based **only** on the uploaded documents.
5. Chat history is displayed above the chat box.

---

## 🏗️ Architecture & Components

| Component           | Tool/Library                                     |
| ------------------- | ------------------------------------------------ |
| Frontend UI         | Streamlit                                        |
| Embeddings          | HuggingFaceEmbeddings                            |
| Vector DB           | Weaviate (Cloud instance)                        |
| LLM                 | Groq API (Meta LLaMA 4 model)                    |
| Document Parsing    | Langchain loaders, PyPDF2, python-docx           |
| Conversation Memory | In-memory chat history (Streamlit session state) |

---

## 📦 Dependencies

* streamlit
* weaviate-client
* python-dotenv
* PyPDF2
* python-docx
* langchain-community
* groq
* huggingface-hub

**Example `requirements.txt`:**

```txt
streamlit==1.34.0
weaviate-client==4.5.4
python-dotenv==1.0.1
PyPDF2==3.0.1
python-docx==1.1.0
langchain-community==0.1.30
groq==0.5.0
huggingface-hub==0.22.2
```

---

## 🔒 Notes

* The chatbot will answer **strictly based on the provided documents**.

* If the answer is not found in the documents, the bot will respond with:

  > *"I don't know based on the provided documents."*

* This ensures transparency and avoids hallucinated answers.

---

## 🛠️ Future Improvements (Ideas)

* Support **multiple documents at once** in chat.
* Add **file upload progress bar**.
* Enable **chat export** (download chat history).
* Add **custom model selector** (choose Groq model at runtime).

---

## 📝 License

MIT License.

---

## 👤 Developer Info

Developed for **Senzmate Interview Demo** by **Gihan Lakmal** 🚀
📧 Email: \[[your-email@example.com](mailto:your-email@example.com)]
💻 GitHub: [your-github-profile](https://github.com/yourusername)

