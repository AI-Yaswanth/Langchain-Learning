from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st

# Page Setup
st.title("Blog -> Quiz Generator")
st.write("Enter an article to generate a quiz.")

# Sidebar Settings
st.sidebar.title("Settings")

# API key input (hidden)
st.session_state.openai_api_key = st.sidebar.text_input(
    label="OpenAI API Key:",
    type="password",
    placeholder="sk-..."
)

# Model selection
st.session_state.model = st.sidebar.selectbox(
    label="Select Model:",
    options=["gpt-4o", "gpt-4o-mini"]
)

# User Input
st.session_state.user_input = st.text_area(
    label="Enter an article:",
    placeholder="e.g. The article content..."
)
button = st.button("Generate Quiz")

# Main Logic: runs only when button is clicked with valid inputs
if button and st.session_state.openai_api_key and st.session_state.user_input:

    # Prompt: summarize the article
    summary_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert article summarizer.

            Your job is to summarize the given article clearly and concisely.

            **RULES:**
            1. Summary must be within 5-7 sentences.
            2. Capture the main idea, key points and conclusion.
            3. Do not add any personal opinions.
            4. Use simple and clear language.
            5. Do not repeat the same point twice.
            """
        ),
        (
            "user",
            "Article: {article}"
        )
    ])
    
    # Prompt: generate quiz questions from the summary
    quiz_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert quiz generator.

            **RULES:**
            1. Generate 5 questions from the summary.
            2. Output questions only - no options, no answers, no explanations.
            3. Questions must be based only on the summary provided."""
        ),
        (
            "user",
            "Summary: {summary}"
        )
    ])

    # Initialize the LLM with user-selected model and API key
    llm = ChatOpenAI(
        model=st.session_state.model,
        api_key=st.session_state.openai_api_key
    )

    # Build chains: prompt -> LLM
    summary_chain = summary_prompt | llm
    quiz_chain = quiz_prompt | llm
    
    # Chain: summary -> quiz
    chain = summary_chain | quiz_chain

    try:
        # Generate summary -> quiz in one chained call
        with st.spinner("Generating Quiz..."):
            result = chain.invoke({
                "article": st.session_state.user_input
            })

        # Display summary
        st.success("Quiz generated successfully!")
        st.subheader("Quiz:")
        st.write(result.content)
    
    except Exception as e:
        st.error(f"Something went wrong: {e}")

# Validation: show warnings if inputs are missing
elif button:
    if not st.session_state.openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    elif not st.session_state.user_input:
        st.warning("Please enter a article to generate a quiz.")
