# import necessary libraries
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Page header
st.title("Prompt Playground")
st.write("Compare how different prompts affect output.")

# Sidebar: API key and model selection
st.sidebar.title("Settings")
st.session_state.openai_api_key = st.sidebar.text_input(label="Enter OpenAI API Key:",type="password")
st.session_state.model = st.sidebar.selectbox(label="Select Model:", options=["gpt-4o", "gpt-4o-mini"])

# Main input: topic the user wants explained
st.session_state.user_input = st.text_input("Enter your question here:")
button = st.button("Explain")

# Run the comparison only when:
# 1. The user has entered a question
# 2. The Explain button is clicked
# 3. An OpenAI API key is provided
if st.session_state.user_input and button and st.session_state.openai_api_key:
    
    # --- Prompt Templates ---
    # Each prompt uses the same user question but different system instructions.
    # This demonstrates how changing the prompt changes the model's behavior and output.
    # The user's question is injected dynamically using {user_input}.
    system_prompt_1 = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Explain the user's question in clear and simple language.

                Your goal is to help someone understand exactly what the user is asking without answering the question.

                Keep the explanation concise and accurate.
                """
            ),
            (
                "user",
                "Here is the user's question: {user_input}"
            )   
        ]
    )
    
    system_prompt_2 = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Explain the user's question as if you are talking to a 10-year-old child.

                Use very simple words, short sentences, and relatable examples.

                Do not answer the question. Only explain what the user is asking.
                """
            ),
            (
                "user",
                "Here is the user's question: {user_input}"
            )
        ]
    )

    system_prompt_3 = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful teacher speaking to a 10-year-old child.

                First, explain what the user's question means using simple everyday language.

                Then answer the question using easy-to-understand explanations and examples.

                Avoid technical jargon.
                """
            ),
            (
                "user",
                "Here is the user's question: {user_input}"
            )
        ]
    )
    
    # --- LLM setup ---
    # Create the OpenAI chat model.
    # The same model is used for all three prompts so that any differences
    # in output come from the prompts rather than the model itself.
    llm = ChatOpenAI(
        model=st.session_state.model,
        api_key=st.session_state.openai_api_key
    )

    # --- Output parser ---
    # Converts the LLM's message object into a plain string
    parser = StrOutputParser()

    # --- Chain ---
    # Build three independent chains:
    # Prompt → Model → Output Parser
    # Each chain uses different instructions but the same model.
    chain_1 = system_prompt_1 | llm | parser
    chain_2 = system_prompt_2 | llm | parser
    chain_3 = system_prompt_3 | llm | parser

    # Execute all three chains using the same user question.
    # The outputs can then be compared side-by-side to observe
    # how prompt wording influences the model's response.
    with st.spinner("Thinking..."):
        response_1 = chain_1.invoke({"user_input": st.session_state.user_input})
        response_2 = chain_2.invoke({"user_input": st.session_state.user_input})
        response_3 = chain_3.invoke({"user_input": st.session_state.user_input})
        
        st.write("--------------------------------")
        st.write("**Prompt 1 Output:**")
        st.write(response_1)
        st.write("--------------------------------")
        st.write("**Prompt 2 Output:**")
        st.write(response_2)
        st.write("--------------------------------")
        st.write("**Prompt 3 Output:**")
        st.write(response_3)
        st.write("--------------------------------")

# Validation messages: only shown after the button is clicked
# so errors don't appear on the initial page load
if button:
    if not st.session_state.openai_api_key:
        st.error("Please enter an OpenAI API Key.")

    elif not st.session_state.user_input:
        st.error("Please enter a question.")
