# import necessary libraries
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Page header
st.title("AI Topic Explainer")
st.write("This app explains the concept of a AI/ML topic in a way that is easy to understand.")

# Sidebar: API key and model selection
st.sidebar.title("Settings")
st.session_state.openai_api_key = st.sidebar.text_input(label="Enter OpenAI API Key:",type="password")
st.session_state.model = st.sidebar.selectbox(label="Select Model:", options=["gpt-4o", "gpt-4o-mini"])

# Main input: topic the user wants explained
st.session_state.user_input = st.text_input("Enter your question here:")
button = st.button("Explain")

# Core logic: only runs when all three conditions are met - user has typed a topic, clicked the button, and provided an API key
if st.session_state.user_input and button and st.session_state.openai_api_key:
    
    # --- Prompt template ---
    # System message defines the assistant's behavior and output structure.
    # User message passes in the topic dynamically via {user_input}.
    system_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an AI education assistant. Your job is to explain AI/ML concepts clearly.

                **CRITICAL RULES:**
                1. FIRST determine if the topic is an AI/ML topic. Topics can include: machine learning, deep learning, NLP, computer vision, reinforcement learning, neural networks, model architectures, training techniques, AI applications, AI safety, AI ethics, AI tools/frameworks, data science concepts used in AI, etc.

                2. If the topic is NOT an AI/ML topic (e.g. cooking, history, sports, politics, general programming unrelated to AI), respond ONLY with:
                "This tool is for AI topics only. '[TOPIC]' doesn't appear to be an AI/ML concept. Try something like 'transformers', 'gradient descent', or 'RAG'."

                3. If it IS an AI topic, respond using this exact structure:

                **Topic:**
                [Topic of the user's question]

                **Definition:**
                [Definition of the topic]

                **Why it is Used:**
                [Why the topic is used]

                **Key Components:**
                [Key components of the topic]

                **Example:**
                [Example of the topic in action]
                
                **Summary:**
                [Summary of the topic]
                """
            ),
            (
                "user",
                "Here is the user's question: {user_input}"
            )
        ]
    )

    # --- LLM setup ---
    # Initializes the OpenAI chat model with the user-selected version and key
    llm = ChatOpenAI(
        model=st.session_state.model,
        api_key=st.session_state.openai_api_key
    )

    # --- Output parser ---
    # Converts the LLM's message object into a plain string
    parser = StrOutputParser()

    # --- Chain ---
    # LangChain pipe: prompt → LLM → parser (returns a clean string)
    chain = system_prompt | llm | parser

    # --- Run the chain and display response ---
    with st.spinner("Thinking..."):
        response = chain.invoke({"user_input": st.session_state.user_input})
        st.write(response)

# Validation messages: only shown after the button is clicked
# so errors don't appear on the initial page load
if button:
    if not st.session_state.openai_api_key:
        st.error("Please enter an OpenAI API Key.")

    elif not st.session_state.user_input:
        st.error("Please enter a question.")
