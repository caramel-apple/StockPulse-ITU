#imports

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
# Swapped CTransformers for ChatOllama
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

#configuration and paths
DOCS_DIR = "./documents"
FAISS_INDEX_PATH = "./faiss_db"
OLLAMA_MODEL_NAME = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF" 
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

#load and index documents
print("[+] Initializing Hugging Face Embeddings...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

#check  if documents are already indexed
if os.path.exists(FAISS_INDEX_PATH):
    print("[+] Loading existing FAISS index from disk...")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
else:
    print("[+] Processing documents from ./docs...")
    
    #load pdf,txt,md
    pdf_loader = DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    md_loader = DirectoryLoader(DOCS_DIR, glob="**/*.md", loader_cls=TextLoader)
    
    documents = pdf_loader.load() + txt_loader.load() + md_loader.load()

    #if no documents
    if not documents:
        print("[!] No documents found in ./docs. Please add .pdf, .txt, or .md files.")
        sys.exit(1)
        
    print(f"[+] Loaded {len(documents)} document page(s). Splitting into chunks...")

    #chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    #vectorizing
    print(f"[+] Generating embeddings and building FAISS index for {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("[+] FAISS index saved successfully!")


#ollama setup
print(f"[+] Connecting to local Ollama model ({OLLAMA_MODEL_NAME})...")

# Initialize ChatOllama (ensure 'pip install langchain-ollama' is run if needed)
llm = ChatOllama(
    model=OLLAMA_MODEL_NAME,
    temperature=0.0,
    num_predict=512,
    num_ctx=2048
)

# systemp prompt
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
        "score_threshold": 0.4  # Adjust between 0.3 - 0.6 based on MiniLM performance
    }
)
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

#cli query loop
#cli query loop (Only run this loop if executing rag.py directly, not when imported)
if __name__ == "__main__":
    print("\n" + "="*50)
    print("   StockPulse RAG Local Assistant Ready!   ")
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