from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st
from pydantic import BaseModel

# Page Setup
st.title("Research Assistant")
st.write("Enter a research topic to generate a structured JSON response.")

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
    label="Enter a research topic:",
    placeholder="e.g. The latest trends in AI..."
)
button = st.button("Generate Research Report")

# Main Logic: runs only when button is clicked with valid inputs
if button and st.session_state.openai_api_key and st.session_state.user_input:

    # Prompt: instruct LLM to return a structured JSON research report
    research_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert research report generator.

            Your job is to generate a structured JSON response based on the given research topic.

            **RULES:**
            1. The response must be in JSON format.
            2. Include these fields: summary, advantages, limitations, future_scope.
            3. advantages, limitations, and future_scope should be lists.

            {formatting_instructions}
            """
        ),
        (
            "user",
            "Research Topic: {research_topic}"
        )
    ])

    # Initialize LLM with user-selected model and API key
    llm = ChatOpenAI(
        model=st.session_state.model,
        api_key=st.session_state.openai_api_key
    )

    # Pydantic model: ensures output is parsed into a valid Python object
    class ResearchReport(BaseModel):
        summary: str
        advantages: list[str]
        limitations: list[str]
        future_scope: list[str]
    
    # JSON parser: ensures output is parsed into a Pydantic object
    parser = JsonOutputParser(pydantic_object=ResearchReport)

    # Chain: prompt -> LLM -> JSON parser
    chain = research_prompt | llm | parser

    try:
        # Invoke chain with topic and formatting instructions
        with st.spinner("Generating Research Report..."):
            result = chain.invoke({
                "research_topic": st.session_state.user_input,
                "formatting_instructions": parser.get_format_instructions()
            })

        # Display the parsed JSON result
        st.success("Research Report generated successfully!")
        st.subheader("Research Report:")
        st.json(result)

    except Exception as e:
        st.error(f"Something went wrong: {e}")

# Validation: show warnings if inputs are missing
elif button:
    if not st.session_state.openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    elif not st.session_state.user_input:
        st.warning("Please enter a research topic to generate a report.")
