# Importing required libraries
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

## Session State
if "files_uploaded" not in st.session_state:
    st.session_state.files_uploaded = []
    
if "n" not in st.session_state:
    st.session_state.n = 0
    
if "show_delete_success" not in st.session_state:
    st.session_state.show_delete_success = False
    
if "vectorstore" not in st.session_state:   
    st.session_state.vectorstore = None

# Guard against a missing/unset API key: assigning None into os.environ raises
# a TypeError, so fail with a clear message instead of crashing on import
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Please add it to your .env file.")
    st.stop()
 
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

## App UI
st.title("Research Paper Semantic Search Engine")

## Configuration
st.sidebar.title("Configuration")
st.session_state.top_k = st.sidebar.slider("Number of results", min_value=1, max_value=10, value=5)

mode = st.radio("Select the mode", ["upload and embed", "Search"])

if mode == "upload and embed":
    uploaded_files = st.file_uploader("Upload a Research Paper PDF file", type="pdf", help="Upload a PDF file to use for the search engine.",accept_multiple_files=True, key=f"uploaded_files_{st.session_state.n}")

    if not uploaded_files:
        st.warning("Please upload at least one file")

    else:
        pending_files = [f for f in uploaded_files if f.name not in st.session_state.files_uploaded]
        for file in uploaded_files:
            if file.name in st.session_state.files_uploaded:
                st.warning(f"**{file.name}** already uploaded and embedded.")

        if pending_files:
            with st.form("upload_form"):
                metadata_inputs = {}

                for file in pending_files:
                    st.markdown(f"Please fill in the metadata details for File: **{file.name}**")
                    title = file.name
                    author = st.text_input("Author", key=f"author_{file.name}")
                    year = st.text_input("Year", key=f"year_{file.name}")

                    metadata_inputs[file.name] = {"title": title, "author": author, "year": year}

                submitted = st.form_submit_button("Upload & Embed All")


            if submitted:
                ## Initialize the splitter and embeddings
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                for file in pending_files:
                    tempfile = f"./temp_{file.name}"

                    try:
                        with st.spinner(f"Embedding {file.name}..."):
                            with open(tempfile, "wb") as f:
                                f.write(file.getvalue())

                            if file.type == "application/pdf":
                                loader = PyPDFLoader(tempfile)
                            
                            else:
                                # Skip the file entirely if it's not a PDF - without this
                                # continue, loader would never be assigned and the
                                # loader.load() call below would raise a NameError.
                                # The uploader is currently PDF-only, so this is a
                                # safety net in case that restriction is ever loosened.
                                st.error(f"Unsupported file type: {file.type}")

                            docs = loader.load()
                            if not docs:
                                st.warning(f"{file.name} contains no readable content.")
                                continue
                            
                            meta = metadata_inputs[file.name]
                            for doc in docs:
                                doc.metadata["title"] = meta["title"]
                                doc.metadata["author"] = meta["author"]
                                doc.metadata["year"] = meta["year"]

                            chunks = splitter.split_documents(docs)

                            if st.session_state.vectorstore is None:
                                st.session_state.vectorstore = Chroma.from_documents(chunks, embeddings)
                            else:
                                st.session_state.vectorstore.add_documents(chunks)

                            st.session_state.files_uploaded.append(file.name)

                    except Exception as e:
                        st.error(f"Error uploading and embedding {file.name}: {e}")

                    finally:
                        if os.path.exists(tempfile):
                            os.remove(tempfile)

                st.success(f"{len(st.session_state.files_uploaded)} document(s) uploaded and tagged.")

    if st.button("Clear Files"):
        st.session_state.n+=1
        st.rerun()

if st.sidebar.button("Delete Files and Embeddings"):
    if not st.session_state.vectorstore is None:
        st.session_state.vectorstore.delete_collection()
        st.session_state.vectorstore = None
        st.session_state.files_uploaded = []
        st.session_state.n = 0
        st.session_state.show_delete_success = True
        st.rerun()
    else:
        st.sidebar.warning("No files uploaded and embedded yet")
    
if st.session_state.get("show_delete_success"):
    st.sidebar.success("Files and embeddings deleted.")
    st.session_state.show_delete_success = False
    
if mode == "Search":
    user_question = st.text_input("Enter your question")
    filter_author = st.text_input("Filter by author (optional)")
    filter_year = st.text_input("Filter by year (optional)")
    
    search_filter = {}
    if filter_author:
        search_filter["author"] = filter_author
    if filter_year:
        search_filter["year"] = filter_year
        
    if st.button("Search"):
        if not user_question:
            st.warning("Please enter a question")
            st.stop()

        if st.session_state.vectorstore is None:
            st.warning("Please upload and embed at least one file")
            st.stop()
            
        else:
            with st.spinner("Thinking..."):
                results = st.session_state.vectorstore.similarity_search_with_score(user_question, k=st.session_state.top_k, filter=search_filter if search_filter else None)

                if not results:
                    st.warning("No results found")
                    st.stop()

                else:
                    with st.spinner("Searching papers..."):
                        results = st.session_state.vectorstore.similarity_search_with_score(
                            user_question, k=st.session_state.top_k
                        )

                    if not results:
                        st.info("No matching papers found.")
                    else:
                        st.markdown(f"### Found {len(results)} results")
                        st.divider()

                        for i, (doc, score) in enumerate(results, start=1):
                            similarity_pct = max(0, min(1, 1 / (1 + score)))  # normalize to 0–1 for display

                            with st.container(border=True):
                                col1, col2 = st.columns([4, 1])

                                with col1:
                                    st.markdown(f"#### {i}. {doc.metadata.get('title', doc.metadata.get('source', 'Untitled'))}")
                                    meta_line = []
                                    if doc.metadata.get("author"):
                                        meta_line.append(f"👤 {doc.metadata.get('author')}")
                                    if doc.metadata.get("year"):
                                        meta_line.append(f"📅 {doc.metadata.get('year')}")
                                    if doc.metadata.get("page") is not None:
                                        meta_line.append(f"📄 Page {doc.metadata.get('page')}")
                                    st.caption(" &nbsp;|&nbsp; ".join(meta_line))

                                with col2:
                                    st.metric("Match", f"{similarity_pct:.0%}")

                                st.progress(similarity_pct)

                                with st.expander("Preview excerpt"):
                                    st.write(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))

                            st.write("")
