from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from pathlib import Path
import os

def load_documents():
    doc_paths = [
        "D:/RAG_Travel_Assistance/rag_travel_assistance/docs/GBERTPaper.pdf",
        "D:/RAG_Travel_Assistance/rag_travel_assistance/docs/GottbertPaper.pdf",
    ]
    
    docs = []
    for doc_file in doc_paths:
        file_path = Path(doc_file)
        try:
            if doc_file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif doc_file.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif doc_file.endswith(".txt") or file_path.name.endswith(".md"):
                loader = TextLoader(file_path)
            else:
                print(f"Document type {file_path.suffix} not supported.")
                continue

            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading document {file_path.name}: {e}")

    # Load web content
    url = "https://www.seatguru.com/airlines/Lufthansa/baggage.php"
    try:
        loader = WebBaseLoader(url)
        url_docs = loader.load()
        docs.extend(url_docs)
    except Exception as e:
        print(f"Error loading document from {url}: {e}")

    return docs

def create_vector_db(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    document_chunks = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return Chroma.from_documents(document_chunks, embeddings)

def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation, focusing on the most recent messages."),
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag_chain(vector_db, llm):
    retriever_chain = _get_context_retriever_chain(vector_db, llm)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are a helpful travel assistant. You will have to answer to user's queries.
        You will have some context to help with your answers, but not always would be completely related or helpful.
        You can also use your knowledge to assist answering the user's queries.\n
        {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_chat_model():
    return ChatOpenAI(
        temperature=0.3,
        model_name="gpt-4",
        streaming=True
    ) 