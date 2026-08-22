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
from langchain_community.llms import CTransformers
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------
# CONFIGURATION & PATHS (Steps 3 & 5)
# ---------------------------------------------------------
DOCS_DIR = "./docs"
FAISS_INDEX_PATH = "./faiss_db"
# Update this filename to match your exact downloaded GGUF model in ./models/
MODEL_PATH = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf" 
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------
# 1. LOAD & INDEX DOCUMENTS (Steps 6, 7 & 8)
# ---------------------------------------------------------
print("[+] Initializing Hugging Face Embeddings...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

if os.path.exists(FAISS_INDEX_PATH):
    print("[+] Loading existing FAISS index from disk...")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
else:
    print("[+] Processing documents from ./docs...")
    
    # Load PDFs, TXTs, and MDs
    pdf_loader = DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    md_loader = DirectoryLoader(DOCS_DIR, glob="**/*.md", loader_cls=TextLoader)
    
    documents = pdf_loader.load() + txt_loader.load() + md_loader.load()
    
    if not documents:
        print("[!] No documents found in ./docs. Please add .pdf, .txt, or .md files.")
        sys.exit(1)
        
    print(f"[+] Loaded {len(documents)} document page(s). Splitting into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    print(f"[+] Generating embeddings and building FAISS index for {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("[+] FAISS index saved successfully!")

# ---------------------------------------------------------
# 2. SETUP LOCAL GGUF LLM (Step 9)
# ---------------------------------------------------------
print("[+] Loading local GGUF model...")
if not os.path.exists(MODEL_PATH):
    print(f"[!] Model file not found at {MODEL_PATH}. Place your .gguf file in ./models/")
    sys.exit(1)

llm = CTransformers(
    model=MODEL_PATH,
    model_type="mistral",  # Change to 'llama' if using a Llama-based GGUF
    config={"max_new_tokens": 512, "temperature": 0.2, "context_length": 2048}
)

# System Prompt Tailored for StockPulse Healthcare
system_prompt = (
    "You are StockPulse AI, an intelligent healthcare supply chain assistant.\n"
    "Use the following pieces of retrieved context to answer the user's question accurately.\n"
    "If you don't know the answer based on the context, say that the information is unavailable in the supply documents.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# ---------------------------------------------------------
# 3. INTERACTIVE CLI QUERY LOOP (Step 10)
# ---------------------------------------------------------
print("\n" + "="*50)
print("  StockPulse RAG Local Assistant Ready!  ")
print("="*50 + "\n")

while True:
    user_query = input("Ask a question about your inventory/documents (or 'exit' to quit): ")
    if user_query.lower() in ["exit", "quit", "q"]:
        break
    if not user_query.strip():
        continue
        
    print("\n[+] Searching vector database and generating response...\n")
    response = rag_chain.invoke({"input": user_query})
    
    print("StockPulse AI Response:")
    print(response["answer"])
    
    print("\n" + "-"*30 + "\nSources Used:")
    for doc in response["context"]:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        print(f" - {source} (Page: {page})")
    print("\n" + "="*50 + "\n")