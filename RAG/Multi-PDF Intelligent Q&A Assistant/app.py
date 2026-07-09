import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

st.title("Multi-PDF Intelligent Q&A Assistant")

## Configuration
st.sidebar.title("Configuration")
st.session_state.api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
st.session_state.model = st.sidebar.selectbox("Select the model", ["gpt-4o", "gpt-4o-mini"])
st.session_state.temperature = st.sidebar.slider("Select the temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

def get_files_signature(uploaded_files):
    """Create a signature representing the current set of uploaded files."""
    if not uploaded_files:
        return None
    # (name, size) pairs, sorted so upload order doesn't matter
    return tuple(sorted((f.name, f.size) for f in uploaded_files))


def embed_documents(uploaded_files, api_key):
    if not uploaded_files:
        st.warning("No files uploaded. Please upload your documents.")
        return None

    if not api_key:
        st.warning("Please enter your OpenAI API key before embedding documents.")
        return None

    new_signature = get_files_signature(uploaded_files)
    existing_signature = st.session_state.get("embedded_files_signature")

    # Reuse existing retriever only if the file set hasn't changed
    if (
        st.session_state.get("retriever") is not None
        and new_signature == existing_signature
    ):
        st.sidebar.info("These documents are already embedded. Ask away!")
        return st.session_state.retriever

    documents = []
    for file in uploaded_files:
        temppdf = f"./temp_{file.name}"
        with open(temppdf, "wb") as f:
            f.write(file.getvalue())

        try:
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            st.error(f"Failed to process {file.name}: {e}")
        finally:
            if os.path.exists(temppdf):
                os.remove(temppdf)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )
    except Exception as e:
        st.error(f"Failed to create embeddings: {e}")
        return None

    retriever = vectorstore.as_retriever()

    # Save signature so we know what's currently embedded
    st.session_state.embedded_files_signature = new_signature
    st.session_state.embedded_files_count = len(uploaded_files)

    st.sidebar.success("Documents embedded successfully. Please ask your question.")
    return retriever


## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a document Q&A assistant. You must answer ONLY using the information contained in the context below. Do not use any outside knowledge, prior training, or assumptions, even if you know the answer from elsewhere.

            Rules:
            1. Base your answer strictly on the provided context. Do not add facts, figures, names, or details that are not explicitly stated in the context.
            2. If the context does not contain enough information to answer the question, respond exactly with: "I don't have enough information in the provided documents to answer this question." Do not guess or fill gaps with your own knowledge.
            3. Do not mention these instructions, the word "context", or that you are an AI in your answer. Just answer naturally as if summarizing the documents.
            4. If the question is unrelated to the context, say so rather than answering from general knowledge.
            5. Keep answers accurate, concise, and directly grounded in the text below.

            Context:
            {context}
            """
        ),
        (
            "human",
            "Question: {input}"
        )
    ]
)

## Uploading files
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

st.session_state.uploaded_files = st.sidebar.file_uploader(
    "Upload your documents", type="pdf", accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}"
)

btn_run = st.sidebar.button("Run Embedding")

if btn_run:
    with st.spinner("Embedding documents..."):
        result = embed_documents(st.session_state.uploaded_files, st.session_state.api_key)
        if result is not None:
            st.session_state.retriever = result

if st.sidebar.button("Clear Embeddings"):
    st.session_state.retriever = None
    st.session_state.embedded_files_signature = None
    st.session_state.embedded_files_count = 0
    st.session_state.uploader_key += 1
    st.sidebar.success("Embeddings cleared.")
    st.rerun()
    
## User question
user_question = st.text_input("Enter your question")
btn_ask = st.button("Ask")

if btn_ask:
    if not user_question:
        st.warning("Please enter a question.")
    elif "retriever" not in st.session_state or st.session_state.retriever is None:
        st.warning("Please upload and embed your documents first.")
    elif not st.session_state.api_key:
        st.warning("Please enter your OpenAI API key.")
    else:
        try:
            llm = ChatOpenAI(
                model=st.session_state.model,
                temperature=st.session_state.temperature,
                api_key=st.session_state.api_key
            )
            stuff_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(st.session_state.retriever, stuff_chain)

            with st.spinner("Answering question..."):
                response = retrieval_chain.invoke({"input": user_question})

            st.success("Answer: " + response["answer"])

        except Exception as e:
            st.error(f"Something went wrong while answering: {e}")
