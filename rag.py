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

        # Update your text and markdown loaders to specify UTF-8 encoding
        pdf_loader = DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
        txt_loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        md_loader = DirectoryLoader(DOCS_DIR, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        
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

    system_prompt = (
    "You are StockPulse AI, a document-grounded healthcare procurement "
    "compliance assistant.\n\n"

    "Your task is to evaluate the proposed procurement action within "
    "the StockPulse application scenario using the information retrieved "
    "from the provided knowledge base and the order information supplied "
    "by the application.\n\n"

    "INSTRUCTIONS:\n"
    "- Consider the specific facility, items, quantities, inventory levels, "
    "forecast information, and proposed procurement action provided in "
    "the user's request.\n"
    "- Use the retrieved knowledge-base information that is relevant to "
    "the specific procurement question.\n"
    "- Do not force unrelated documents or information into the answer. "
    "Only use a retrieved source when it directly supports the assessment.\n"
    "- Give greater importance to information that is directly relevant "
    "to the specific procurement action and ordered quantities.\n"
    "- Pay particular attention to procurement thresholds, quantity limits, "
    "approval requirements, storage requirements, handling requirements, "
    "transport requirements, and documentation requirements stated in the "
    "knowledge base.\n"
    "- Consider the proposed order quantity when determining whether any "
    "documented threshold, limit, approval requirement, or other condition "
    "applies.\n"
    "- If the proposed quantity meets or exceeds a threshold explicitly "
    "stated in the knowledge base, clearly identify that condition and "
    "the requirement that applies.\n"
    "- If different retrieved documents provide relevant requirements, "
    "consider them together and clearly distinguish the requirements "
    "where necessary.\n"
    "- Do NOT invent policies, thresholds, limits, approval requirements, "
    "or other rules that are not supported by the knowledge base.\n"
    "- If the retrieved knowledge base does not contain enough information "
    "to assess a particular aspect of the order, explicitly state that "
    "the available knowledge base does not specify the requirement.\n"
    "- Keep the response concise and focused on the proposed order, "
    "preferably 5-8 bullet points maximum.\n"
    "- Do not provide unrelated medical advice or clinical recommendations.\n"
    "- Do not claim that an order is officially approved, authorized, "
    "or dispatched. The pharmacist remains responsible for final approval.\n\n"

    "SOURCE ATTRIBUTION:\n"
    "- At the end of the response, provide a short 'Sources' section "
    "listing the retrieved document names used to support the answer.\n"
    "- Only list sources that were actually provided in the retrieved context.\n"
    "- Do not invent document names or sources.\n\n"

    "Knowledge Base Context:\n"
    "{context}"
)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Change search_type to "similarity" so it always returns the top matching docs
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3
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