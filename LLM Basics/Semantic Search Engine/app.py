# Importing the necessary libraries
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# UI Configuration

st.title("Semantic Search Engine")
st.sidebar.title("Settings")

# Session State Initialization
def initialize_session_state():
    """Initialize required session state variables."""
    if "embedding_status" not in st.session_state:
        st.session_state.embedding_status = False
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None

initialize_session_state()

# Sidebar Configuration
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
model = st.sidebar.selectbox("Select a model", ["gpt-4o", "gpt-4o-mini"])

uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")
process_button = st.sidebar.button("Process File")

# File Processing Function

def process_pdf_file(uploaded_file, api_key):
    """
    Process uploaded PDF file and create vector embeddings.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        api_key: OpenAI API key for embeddings
    """
    try:
        # Create temporary file from uploaded content
        temp_pdf_path = "./temp.pdf"
        with open(temp_pdf_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getvalue())
        
        # Load and split PDF into chunks
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Clean up temporary file
        os.remove(temp_pdf_path)
        
        return vectorstore, vectorstore.as_retriever()
    
    except Exception as e:
        st.sidebar.error(f"Error processing PDF: {str(e)}")
        return None, None

# PDF Processing Logic
if process_button:
    # Validate inputs before processing
    if not api_key:
        st.sidebar.warning("⚠️ Please enter your OpenAI API key")
    elif not uploaded_file:
        st.sidebar.warning("⚠️ Please upload a PDF file")
    elif not st.session_state.embedding_status:
        # Process the file
        with st.spinner("Processing PDF and creating embeddings..."):
            vectorstore, retriever = process_pdf_file(uploaded_file, api_key)
            
            if vectorstore is not None:
                st.session_state.vectorstore = vectorstore
                st.session_state.retriever = retriever
                st.session_state.embedding_status = True
                st.sidebar.success("✅ PDF processed successfully!")

# Search Logic
if st.session_state.embedding_status:
    # Display search interface
    user_query = st.text_input("Enter your search query")
    search_button = st.button("Search")
    
    if user_query and search_button:
        with st.spinner("Searching and generating response..."):
            try:
                # Retrieve relevant documents
                retrieved_docs = st.session_state.retriever.invoke(user_query)
                
                # Initialize language model
                llm = ChatOpenAI(api_key=api_key, model=model, temperature=0.7)
                
                # Create prompt template with system and user messages
                prompt_template = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        "You are a helpful assistant specialized in answering questions based on "
                        "provided documents. Use the retrieved context to answer the user's question. "
                        "If the answer is not in the context, say 'I don't know'. "
                        "Provide the answer in 3 concise bullet points.\n\n"
                        "Context:\n{context}"
                    ),
                    (
                        "human",
                        "{input}"
                    )
                ])
                
                # Create and execute the RAG chain
                chain = prompt_template | llm | StrOutputParser()
                response = chain.invoke({
                    "context": retrieved_docs,
                    "input": user_query
                })
                
                # Display results
                st.subheader("Search Results")
                st.write(response)
                
            except Exception as e:
                st.error(f"Error during search: {str(e)}")
    
    elif not user_query and search_button:
        st.warning("⚠️ Please enter a search query before clicking search")

else:
    # Display message when PDF hasn't been processed yet
    st.warning("📄 Please process a PDF file first to enable semantic search")
