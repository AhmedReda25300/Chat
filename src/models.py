import os
import time
import fitz  # PyMuPDF for PDF processing
from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document, AIMessage, HumanMessage
import streamlit as st

# Load environment variables
load_dotenv()

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    text = ""
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

def get_vectorstore_from_url_or_pdfs(user_input, files=None):
    """Create a vector store from URL or multiple PDFs."""
    try:
        google_embedding_model = os.getenv("GOOGLE_EMBEDDING_MODEL")
        if not google_embedding_model:
            raise ValueError("Google Embedding Model is not set in environment variables.")
        
        documents = []

        if files:
            # Process multiple PDF files
            for file in files:
                text = extract_text_from_pdf(file)
                documents.append(Document(page_content=text))
        elif user_input:
            # Process URL
            loader = WebBaseLoader(user_input)
            documents = loader.load()
            if not documents:
                raise ValueError("No documents were loaded from the URL.")

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(documents)

        # Extract text and create embeddings
        texts = [doc.page_content for doc in document_chunks]
        embeddings = GoogleGenerativeAIEmbeddings(model=google_embedding_model)
        vector_store = FAISS.from_texts(texts=texts, embedding=embeddings)

        return vector_store

    except Exception as e:
        st.error(f"An error occurred while creating the vector store: {e}")
        return None

def get_context_retriever_chain(vector_store):
    """Create a context-aware retriever chain."""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        retriever = vector_store.as_retriever()
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])

        # Create and return the retriever chain
        retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
        return retriever_chain

    except Exception as e:
        st.error(f"An error occurred while creating the retriever chain: {e}")
        return None

def get_conversation_rag_chain(retriever_chain):
    """Create the conversation RAG chain."""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        # Create the documents chain
        stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
        
        return create_retrieval_chain(retriever_chain, stuff_documents_chain)

    except Exception as e:
        st.error(f"An error occurred while creating the conversation RAG chain: {e}")
        return None

def get_response(user_input):
    """Generate a response based on user input."""
    try:
        if 'vector_store' not in st.session_state:
            raise ValueError("Vector store is not set. Please upload a PDF or enter a URL.")

        retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
        if not retriever_chain:
            raise ValueError("Retriever chain could not be created.")

        conversation_rag_chain = get_conversation_rag_chain(retriever_chain)
        if not conversation_rag_chain:
            raise ValueError("Conversation RAG chain could not be created.")

        input_data = {
            "chat_history": st.session_state.chat_history,
            "input": user_input
        }

        for attempt in range(3):  # Retry up to 3 times
            try:
                response = conversation_rag_chain.invoke(input_data)
                if 'answer' not in response:
                    raise KeyError("Response does not contain 'answer' key.")
                return response['answer']
            except InternalServerError:
                if attempt < 2:
                    time.sleep(2)  # Wait before retrying
                    continue
                else:
                    raise
            except ResourceExhausted:
                st.error("Resource quota exceeded. Please try again later.")
                return "Sorry, resource quota exceeded. Please try again later."

    except Exception as e:
        st.error(f"An error occurred while getting the response: {e}")
        return "Sorry, an error occurred while processing your request."
