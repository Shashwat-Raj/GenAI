import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import tempfile

# Set your OpenAI API Key securely
OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

def create_vectorstore(documents):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documents, embeddings)

def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# Streamlit UI
st.set_page_config(page_title="Document Chatbot with RAG", layout="wide")
st.title("📄💬 Chat with Your Document (LangChain + RAG)")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("📄 Document uploaded and processing...")
    docs = load_pdf(tmp_path)
    vectorstore = create_vectorstore(docs)
    qa_chain = create_qa_chain(vectorstore)
    st.success("✅ Chatbot is ready!")

    # Chat UI
    user_question = st.text_input("Ask a question based on the document:")
    if user_question:
        with st.spinner("Thinking..."):
            response = qa_chain.run(user_question)
        st.markdown(f"**Answer:** {response}")
