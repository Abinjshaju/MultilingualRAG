import os
import re
import tempfile
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import pdfplumber
import streamlit as st

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.callbacks.base import BaseCallbackHandler

CHROMA_PATH = "./chroma_agent_db"
HF_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL = "gemini-2.0-flash-exp"
EXAMPLE_PDF_PATH = "./docs/example.pdf"

def extract_text_from_pdf(pdf_path: str) -> str:
    extracted_text = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text.append(text)
                else:
                    img = page.to_image(resolution=300).original
                    ocr_text = pytesseract.image_to_string(img, lang="eng+mal")
                    extracted_text.append(ocr_text)
    except Exception as e:
        images = convert_from_path(pdf_path)
        for img in images:
            ocr_text = pytesseract.image_to_string(img, lang="eng+mal")
            extracted_text.append(ocr_text)

    return "\n".join(extracted_text)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=700, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def create_vectorstore(chunks):
    embedding_model = SentenceTransformer(HF_EMBEDDING_MODEL)
    embeddings = SentenceTransformerEmbeddings(model_name=HF_EMBEDDING_MODEL)
    vectorstore = Chroma.from_texts(chunks, embedding=embeddings, persist_directory=CHROMA_PATH)
    vectorstore.persist()
    return vectorstore

class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.logs = ""

    def on_llm_new_token(self, token, **kwargs):
        self.logs += token
        self.container.markdown(f"** Agent Thought:** {self.logs}")

    def on_chain_end(self, outputs, **kwargs):
        self.container.markdown(f"** Final Thought:** {self.logs}")

def create_agent(vectorstore, thought_container, gemini_api_key: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=gemini_api_key,
        temperature=0.2
    )

    def retrieve_docs(query):
        results = retriever.get_relevant_documents(query)
        if results:
            thought_container.markdown("### Retrieved Documents")
        for i, doc in enumerate(results):
            with st.expander(f"Result {i+1}"):
                st.write(doc.page_content[:500] + "...")
        return "\n\n".join([doc.page_content for doc in results])

    tools = [
        Tool(
            name="Document Retriever",
            func=retrieve_docs,
            description="Retrieve relevant context from the processed documents"
        )
    ]

    callbacks = [StreamlitCallbackHandler(thought_container)]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        callbacks=callbacks
    )
    return agent

st.set_page_config(page_title=" Agentic RAG with LangChain + ChromaDB", layout="wide")

st.title(" Agentic Retrieval-Augmented Generation (RAG)")
st.caption("Supports Malayalam + English | Handwritten + Scanned PDFs | Powered by LangChain + Gemini")

st.sidebar.header(" API Configuration")
gemini_api_key = st.sidebar.text_input("Google Gemini API Key", type="password")

st.sidebar.header(" Document Setup")
uploaded_file = st.sidebar.file_uploader("Upload PDF (Scanned or Normal)", type=["pdf"])
use_example = st.sidebar.checkbox("Use built-in example.pdf", value=True)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name
elif use_example and os.path.exists(EXAMPLE_PDF_PATH):
    pdf_path = EXAMPLE_PDF_PATH
else:
    pdf_path = None

if pdf_path:
    st.sidebar.success(f"Using: {os.path.basename(pdf_path)}")
else:
    st.sidebar.error("Please upload or enable example PDF.")

if pdf_path and st.button(" Process & Embed PDF"):
    with st.spinner("Extracting text and creating embeddings..."):
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            st.error(" No text extracted. Possibly unreadable file.")
        else:
            clean = clean_text(text)
            chunks = chunk_text(clean)
            vectorstore = create_vectorstore(chunks)
            st.session_state["vectorstore"] = vectorstore
            st.success(f" Document processed and stored ({len(chunks)} chunks).")

st.divider()
st.subheader(" Ask your document an intelligent question")
query = st.text_area("Enter your query (Malayalam or English):", height=100)

if st.button(" Run Agentic Query"):
    if not gemini_api_key:
        st.warning("Please enter your Google Gemini API Key in the sidebar.")
    elif "vectorstore" not in st.session_state:
        st.warning("Please process a document first.")
    elif not query.strip():
        st.warning("Please enter a query.")
    else:
        thought_container = st.empty()
        agent = create_agent(st.session_state["vectorstore"], thought_container, gemini_api_key)
        with st.spinner(" Agent reasoning and retrieving context..."):
            response = agent.run(query)
        st.markdown("### Agent Response")
        st.write(response)
