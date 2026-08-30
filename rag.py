import os
import sys
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Configuration and paths
DOCS_DIR = "./documents"
FAISS_INDEX_PATH = "./faiss_db"
OLLAMA_MODEL_NAME = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF" 
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def build_rag_chain():
    # Load and index documents
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Check if documents are already indexed
    if os.path.exists(FAISS_INDEX_PATH):
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        if not os.path.exists(DOCS_DIR):
            os.makedirs(DOCS_DIR)

        pdf_loader = DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
        txt_loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
        md_loader = DirectoryLoader(DOCS_DIR, glob="**/*.md", loader_cls=TextLoader)
        
        documents = pdf_loader.load() + txt_loader.load() + md_loader.load()

        if not documents:
            # Fallback text if folder is empty so Streamlit doesn't crash on boot
            vectorstore = FAISS.from_texts(
                ["Standard healthcare supply chain safety guidelines context."], 
                embeddings
            )
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(FAISS_INDEX_PATH)

    # Ollama setup
    llm = ChatOllama(
        model=OLLAMA_MODEL_NAME,
        temperature=0.0,
        num_predict=512,
        num_ctx=2048
    )

    # System prompt
    system_prompt = (
        "You are StockPulse AI, an intelligent healthcare supply chain assistant.\n"
        "Strictly use the following pieces of retrieved content from the supply documents to answer the user's question accurately.\n"
        "If the user's question is not relevant to the supply documents, output EXACTLY:\n"
        "\"The information is unavailable in the supply documents.\"\n"
        "If you don't know the answer based on the suppy documents, say that the information is unavailable in the supply documents and do not try to further elaborate.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.4
        }
    )

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)

# Initialize single chain instance for main.py import
try:
    rag_chain = build_rag_chain()
except Exception as e:
    rag_chain = None
    print(f"[!] Warning: RAG chain initialization failed: {e}")