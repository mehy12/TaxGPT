import streamlit as st
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ✅ Streamlit Page Config
st.set_page_config(page_title="🧾 TaxGPT - Ask your Tax & Audit Questions", layout="centered")
st.title("🧾 TaxGPT - Ask your Tax & Audit Questions")

# 🔐 API Key
os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]

# 📄 Upload PDF
uploaded_file = st.file_uploader("Upload a CA-related PDF (e.g., GST Act)", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyMuPDFLoader("temp.pdf")
    pages = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = FAISS.from_documents(docs, embeddings)

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(),
        memory=memory
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_q = st.text_input("Ask your question:")

    if user_q:
        response = qa.run(user_q)
        st.session_state.chat_history.append((user_q, response))

    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")

else:
    st.info("📄 Please upload a PDF to begin chatting.")
