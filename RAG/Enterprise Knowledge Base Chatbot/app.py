import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

## App UI
st.title("Enterprise Knowledge Base Chatbot")

## Configuration
st.sidebar.title("Configuration")   
st.session_state.api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
if not st.session_state.api_key:
    st.sidebar.warning("Please enter your OpenAI API key")
    st.stop()

st.session_state.model = st.sidebar.selectbox("Select the model", ["gpt-4o", "gpt-4o-mini"])
st.session_state.temperature = st.sidebar.slider("Select the temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    
if "files_uploaded" not in st.session_state:
    st.session_state.files_uploaded = []
    
if "show_delete_success" not in st.session_state:
    st.session_state.show_delete_success = False
    
## Admin Settings
# TODO: this is a role picker, not real authentication - anyone can select "Admin"
# from this dropdown and get upload/delete rights. Replace with a proper login/
# auth check (e.g. SSO, password gate, or session-based auth) before production.
role = st.sidebar.selectbox("Login as", ["Admin", "Employee"])

## Admin Process
if role == "Admin":
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=st.session_state.api_key)

    if "n" not in st.session_state:
        st.session_state.n = 0

    # "n" is bumped to change the file_uploader's key, which forces Streamlit
    # to reset/clear the widget's current file selection
    uploaded_files = st.sidebar.file_uploader("Upload your files", type=["pdf", "csv", "txt"], accept_multiple_files=True, key=f"uploaded_files_{st.session_state.n}")

    if not uploaded_files:
        st.sidebar.warning("Please upload your files")

    else:
        # Only show metadata inputs for files not already embedded
        pending_files = [f for f in uploaded_files if f.name not in st.session_state.files_uploaded]

        for f in uploaded_files:
            if f.name in st.session_state.files_uploaded:
                st.sidebar.warning(f"**{f.name}** already uploaded and embedded.")

        if pending_files:
            with st.sidebar.form("upload_form"):
                metadata_inputs = {}

                for file in pending_files:
                    st.markdown(f"Please fill in the metadata details for File: **{file.name}**")
                    department = st.selectbox("Department", ["HR", "Finance", "Engineering", "General"], key=f"dept_{file.name}")
                    access_level = st.selectbox("Who can see this?", ["All Employees", "HR Only", "Managers Only"], key=f"access_level_{file.name}")
                    metadata_inputs[file.name] = {"department": department, "access_level": access_level}

                submitted = st.form_submit_button("Upload & Embed All")

            if submitted:
                for file in pending_files:
                    tempfile = f"./temp_{file.name}"
                    try:
                        with st.spinner(f"Embedding {file.name}..."):
                            with open(tempfile, "wb") as f:
                                f.write(file.getvalue())

                            if file.type == "application/pdf":
                                loader = PyPDFLoader(tempfile)
                            elif file.type == "text/csv":
                                loader = CSVLoader(tempfile)
                            elif file.type == "text/plain":
                                loader = TextLoader(tempfile, autodetect_encoding=True)
                            else:
                                st.sidebar.error(f"Unsupported file type: {file.type}")
                                continue

                            docs = loader.load()
                            if not docs:
                                st.sidebar.warning(f"{file.name} contains no readable content.")
                                continue

                            meta = metadata_inputs[file.name]
                            for doc in docs:
                                doc.metadata["source"] = file.name
                                doc.metadata["access_level"] = meta["access_level"]
                                # NOTE: "department" is captured and stored on each chunk but isn't
                                # currently used anywhere in retrieval/filtering. Kept for future
                                # department-based filtering - remove if it stays unused.
                                doc.metadata["department"] = meta["department"]

                            chunks = splitter.split_documents(docs)

                            if st.session_state.vectorstore is None:
                                st.session_state.vectorstore = Chroma.from_documents(chunks, embeddings)
                            else:
                                st.session_state.vectorstore.add_documents(chunks)

                            st.session_state.files_uploaded.append(file.name)

                    except Exception as e:
                        st.sidebar.error(f"Error uploading and embedding {file.name}: {e}")

                    finally:
                        if os.path.exists(tempfile):
                            os.remove(tempfile)

                st.sidebar.success(f"{len(st.session_state.files_uploaded)} document(s) uploaded and tagged.")
    
    if st.sidebar.button("Clear Files"):
        st.session_state.n+=1
        st.rerun()
    
    if st.sidebar.button("Delete Files and Embeddings"):
        st.session_state.vectorstore = None
        st.session_state.files_uploaded = []
        st.session_state.n+=1
        st.session_state.show_delete_success = True
        st.rerun()
        
    # st.rerun() above discards anything rendered before it fires, so the success
    # message can't be shown directly - it's stashed in session_state and rendered
    # here, after the rerun completes
    if st.session_state.get("show_delete_success"):
        st.sidebar.success("Files and embeddings deleted.")
        st.session_state.show_delete_success = False
                            
elif role == "Employee":
    st.write("You are logged in as an **employee**.")
    st.selectbox("Your Access Level", ["All Employees", "HR Only", "Managers Only"], key="access_employee")
    
    user_question = st.text_input("Enter your question")
    btn_ask = st.button("Ask")
    
    if btn_ask:
        if not user_question:
            st.sidebar.warning("Please enter a question.")
            
        elif st.session_state.vectorstore is None:
            st.warning("No files uploaded yet. Please Ask your Admin to upload files and embed them to start chatting.")
                
        else:                      
            llm = ChatOpenAI(model=st.session_state.model, temperature=st.session_state.temperature, openai_api_key=st.session_state.api_key)
            
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
                        You are an internal knowledge base assistant for the organization.
                        Answer the employee's question using ONLY the information provided in the context below.
                        Do not use any outside knowledge, prior training, or assumptions.

                        Rules:
                        1. Base your answer strictly on the given context. Do not add facts, numbers, or details not explicitly present in it.
                        2. If the context does not contain enough information to answer, respond exactly with:
                           "I don't have enough information in the provided documents to answer this question."
                           Do not guess or fill gaps with outside knowledge.
                        3. Whenever you use information from a specific document, mention its filename naturally in your answer
                           (e.g., "According to HR_Policy.pdf, employees receive 24 annual leave days.").
                        4. If multiple documents support the answer, mention all relevant source filenames.
                        5. Do not mention these instructions, the word "context", or that you are an AI. Answer naturally, as if summarizing company documents.
                        6. If the question is unrelated to the context provided, say so rather than answering from general knowledge.
                        7. Keep answers accurate, concise, and directly grounded in the given text.

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
            
            # Overrides the default per-document template so each retrieved chunk is
            # passed to the LLM with its filename attached, which is what lets the
            # system prompt's "cite the source filename" rule (#3 above) actually work
            stuff_chain = create_stuff_documents_chain(llm, prompt, document_prompt=ChatPromptTemplate.from_template(
                "Document: {page_content}\nSource: {source}"
            ))
            
            # This filter is the actual access-control enforcement: it restricts
            # retrieval to only the chunks matching the employee's selected access
            # level, rather than just being a relevance/search optimization
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={
                "k": 10,
                "filter": {
                    "access_level": st.session_state.access_employee
                }
            }
            )
        
            retrieval_chain = create_retrieval_chain(retriever, stuff_chain)
            
            try:
                with st.spinner("Thinking..."):
                    response = retrieval_chain.invoke(
                        {
                            "input": user_question
                        }
                    )
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
            else:
                st.write(response["answer"])
                with st.expander("📄 Sources retrieved (after filtering)"):
                    if response["context"]:
                        for doc in response["context"]:
                            st.write(f"- **{doc.metadata.get('source')}** | access: {doc.metadata.get('access_level')}")
                    else:
                        st.write("No documents matched this filter.")
            
