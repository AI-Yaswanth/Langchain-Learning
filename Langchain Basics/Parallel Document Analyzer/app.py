# Import necessary libraries
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_core.runnables import RunnableParallel

st.title("Parallel Document Analyzer")
st.write("Run multiple analyses simultaneously.")

# Sidebar: Settings
st.sidebar.title("Settings")
st.session_state.openai_api_key = st.sidebar.text_input(
    label="OpenAI API Key:",
    type="password",
    placeholder="sk-..."
)
st.session_state.model = st.sidebar.selectbox(
    label="Select Model:",
    options=["gpt-4o", "gpt-4o-mini"]
)

# Main Input
st.session_state.user_input = st.text_input(
    label="Document Text",
    placeholder="e.g. Transformers, RAG, Gradient Descent..."
)
button = st.button("Analyze Document")

# Run parallel analysis when button is clicked
if button and st.session_state.openai_api_key and st.session_state.user_input:
    Summary_prompt = ChatPromptTemplate.from_messages(
        [("system", "You are an helpful assitant. Your task is to summarize the following document in 3 bullet points: {document}")]
    )
    
    Keywords_prompt = ChatPromptTemplate.from_messages(
        [("system", "You are an helpful assitant. Extract 5 important keywords from this document. Return only comma-separated keywords. {document}")]
    )
    
    Sentiment_prompt = ChatPromptTemplate.from_messages(
        [("system", "You are an helpful assitant. Analyze the sentiment of this document. Return only one word: Positive, Negative, or Neutral. {document}")]
    )
    
    Difficulty_prompt = ChatPromptTemplate.from_messages(
        [("system", "You are an helpful assitant. Your task is to classify the reading difficulty of this document. Return only one word: Beginner, Intermediate, or Advanced. {document}")]
    )

    llm = ChatOpenAI(
        api_key=st.session_state.openai_api_key,
        model=st.session_state.model
    )

    # Build individual chains
    summary_chain = Summary_prompt | llm | StrOutputParser()
    keywords_chain = Keywords_prompt | llm | StrOutputParser()
    sentiment_chain = Sentiment_prompt | llm | StrOutputParser()
    difficulty_chain = Difficulty_prompt | llm | StrOutputParser()

    # Combine into a parallel chain
    parallel_chain = RunnableParallel({
        "summary":    summary_chain,
        "keywords":   keywords_chain,
        "sentiment":  sentiment_chain,
        "difficulty": difficulty_chain
    })

    try:
        with st.spinner("Analyzing Document Text..."):
            result = parallel_chain.invoke({"document": st.session_state.user_input})
            st.write(result)
    except Exception as e:
        st.error(f"Something went wrong: {e}")

# Show warnings if inputs are missing
elif button:
    if not st.session_state.openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    elif not st.session_state.user_input:
        st.warning("Please enter a document text to analyze.")
