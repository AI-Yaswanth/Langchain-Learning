# IMPORTS
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# SESSION STATE INITIALIZATION
# Track uploaded file names to prevent duplicate uploads
if "files_uploaded" not in st.session_state:
    st.session_state.files_uploaded = []
    
# Counter used to refresh the file uploader widget
if "n" not in st.session_state:
    st.session_state.n = 0
    
# Flag to show delete success message once, then hide it
if "show_delete_success" not in st.session_state:
    st.session_state.show_delete_success = False
    
# Store the Chroma vector database in session state for persistence across reruns
if "vectorstore" not in st.session_state:   
    st.session_state.vectorstore = None

# LLM AND PROMPT CONFIGURATION
# Initialize the language model (GPT-4o Mini for cost efficiency)
llm = ChatOpenAI(model="gpt-4o-mini")

# Define the system prompt and structure for the RAG chain
prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful assistant that can answer questions based on the provided context.

                Answer the user's question using only the supplied context.

                **Rules:**
                    1. Do not use outside knowledge.
                    2. Do not invent information.
                    3. If the context does not contain enough information, respond exactly: "I don't know based on the provided documents."
                    4. Keep the answer clear and concise.
                    5. Mention the source and page number when available.

                **Context:**
                {context}
                """
            ),
            ("human", "{question}"),
        ]
    )

# Create the RAG chain: prompt → LLM → string output parser
chain = prompt | llm | StrOutputParser()

# API KEY AND CONFIGURATION
# Get OpenAI API key from user input in sidebar
st.session_state.api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
if not st.session_state.api_key:
    st.sidebar.error("API key is not set. Please enter it above.")
    st.stop()
 
# Set the API key as an environment variable for OpenAI client
os.environ["OPENAI_API_KEY"] = st.session_state.api_key

# Allow user to configure retrieval parameters
# top_k: number of most relevant chunks to retrieve
st.session_state.top_k = st.sidebar.number_input("Enter the number of results to return", value=3, min_value=1, max_value=10, step=1)

# score_threshold: minimum similarity score to include a result (0-1, higher = more strict)
st.session_state.score_threshold = st.sidebar.number_input("Enter the score threshold", value=0.65, min_value=0.0, max_value=1.0, step=0.05)

# APP UI - MAIN TITLE AND MODE SELECTION
st.title("I Don't Know RAG Bot")
st.markdown("Upload PDF documents and ask questions about their content. This RAG bot retrieves relevant information from your documents and provides accurate answers.")

st.sidebar.title("Configuration")
# Radio button to switch between upload and search modes
mode = st.radio("Select the mode", ["upload and embed", "Search"])

# MODE 1: UPLOAD AND EMBED
if mode == "upload and embed":
    # File uploader widget - key includes counter to refresh after clearing files
    uploaded_files = st.file_uploader("Upload a PDF file", type="pdf", help="Upload a PDF file for embeddings.",accept_multiple_files=True, key=f"uploaded_files_{st.session_state.n}")

    if not uploaded_files:
        st.warning("Please upload at least one file")

    else:
        # Filter out files that have already been uploaded and embedded
        pending_files = [f for f in uploaded_files if f.name not in st.session_state.files_uploaded]
        
        # Show warning for files that are already embedded
        for file in uploaded_files:
            if file.name in st.session_state.files_uploaded:
                st.warning(f"**{file.name}** already uploaded and embedded.")

        if pending_files:
            if st.button("Upload & Embed All"):
                # Initialize text splitter: chunk_size=500 chars, 50 char overlap
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                # Initialize embeddings model for vectorization
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                
                # Process each pending file
                for file in pending_files:
                    # Create temporary file path
                    tempfile = f"./temp_{file.name}"

                    try:
                        with st.spinner(f"Embedding {file.name}..."):
                            # Write uploaded file to temporary location
                            with open(tempfile, "wb") as f:
                                f.write(file.getvalue())

                            # Load PDF content
                            if file.type == "application/pdf":
                                loader = PyPDFLoader(tempfile)
                            
                            else:
                                # Safety check: only PDFs are supported by the file uploader
                                st.error(f"Unsupported file type: {file.type}")
                                continue

                            # Extract text and metadata from PDF
                            docs = loader.load()
                            if not docs:
                                st.warning(f"{file.name} contains no readable content.")
                                continue

                            # Split documents into chunks for better retrieval
                            chunks = splitter.split_documents(docs)

                            # Add chunks to vectorstore (create new one if it doesn't exist)
                            if st.session_state.vectorstore is None:
                                st.session_state.vectorstore = Chroma.from_documents(chunks, embeddings)
                            else:
                                st.session_state.vectorstore.add_documents(chunks)

                            # Track that this file has been successfully processed
                            st.session_state.files_uploaded.append(file.name)

                    except Exception as e:
                        st.error(f"Error uploading and embedding {file.name}: {e}")

                    finally:
                        # Clean up temporary file regardless of success or failure
                        if os.path.exists(tempfile):
                            os.remove(tempfile)

                st.success(f"{len(st.session_state.files_uploaded)} document(s) uploaded and tagged.")

    # Button to clear the file uploader widget by incrementing the counter
    if st.button("Clear Files"):
        st.session_state.n+=1
        st.rerun()

# DELETE EMBEDDINGS (Available in sidebar regardless of mode)
if st.sidebar.button("Delete Files and Embeddings"):
    if not st.session_state.vectorstore is None:
        # Delete all stored embeddings and reset session state
        st.session_state.vectorstore.delete_collection()
        st.session_state.vectorstore = None
        st.session_state.files_uploaded = []
        st.session_state.n = 0
        st.session_state.show_delete_success = True
        st.rerun()
    else:
        st.sidebar.warning("No files uploaded and embedded yet")
    
# Show deletion success message once, then clear the flag
if st.session_state.get("show_delete_success"):
    st.sidebar.success("Files and embeddings deleted.")
    st.session_state.show_delete_success = False

# MODE 2: SEARCH
if mode == "Search":
    # Text input for user question
    user_question = st.text_input("Enter your question")
    
    if st.button("Search"):
        # Validation: ensure question is provided
        if not user_question:
            st.warning("Please enter a question")
            st.stop()

        # Validation: ensure files have been embedded
        if st.session_state.vectorstore is None:
            st.warning("Please upload and embed at least one file")
            st.stop()

        with st.spinner("Thinking..."):
            # Create a retriever using similarity score threshold strategy
            # This ensures only relevant chunks above the threshold are returned
            threshold_result = st.session_state.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": st.session_state.top_k, "score_threshold": st.session_state.score_threshold}
            )
            
            # Retrieve relevant context chunks based on the question
            threshold_result = threshold_result.invoke(user_question)
        
            # Generate response using the RAG chain with retrieved context
            response = chain.invoke({"context": threshold_result, "question": user_question})
            st.write(response)
            
            st.divider()
            # Display the context chunks that were used to generate the response
            if len(threshold_result) > 0:
                st.write("Here is the context for the score threshold retrieval strategy:")
                for res in threshold_result:
                    st.write(res.page_content)
                    st.divider()
