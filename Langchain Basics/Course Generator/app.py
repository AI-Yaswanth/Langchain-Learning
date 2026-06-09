# Import necessary libraries
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
import streamlit as st
from typing import Optional

# Page Config
st.title("AI/ML Course Generator")
st.write("Enter an AI/ML topic to generate a structured course outline.")

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
    label="Enter an AI/ML topic:",
    placeholder="e.g. Transformers, RAG, Gradient Descent..."
)
button = st.button("Generate Course")

# Main Logic
if button and st.session_state.openai_api_key and st.session_state.user_input:

    # Output Schema
    class Details(BaseModel):
        course_name: Optional[str] = None       # Full course title
        difficulty: Optional[int] = None         # 1 (Beginner) to 5 (Expert)
        duration: Optional[str] = None           # Estimated completion time
        modules: Optional[list[str]] = None      # List of course modules
        is_ai_topic: bool = False                # True if valid AI/ML topic
        error_message: Optional[str] = None      # Populated if not AI/ML topic

    # Parser
    # Create parser once and reuse for both chain and format instructions
    parser = PydanticOutputParser(pydantic_object=Details)

    # Prompt Template
    system_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an AI/ML education assistant that generates structured course outlines.

            **CRITICAL RULES:**
            1. FIRST determine if the topic is an AI/ML topic.
               - AI/ML topics include: machine learning, deep learning, NLP,
                 neural networks, computer vision, transformers, RAG,
                 LLMs, reinforcement learning, AI frameworks, etc.
               - Non AI/ML topics include: cooking, sports, history,
                 general programming unrelated to AI, etc.

            2. If NOT an AI/ML topic, return:
               {{"is_ai_topic": false, "error_message": "Not an AI/ML topic"}}

            3. If it IS an AI/ML topic, return a fully structured course outline
               with course name, difficulty (1-5), duration, and modules list.

            {format_instructions}"""
        ),
        (
            "user",
            "Topic: {topic}"
        )
    ])

    # LLM
    llm = ChatOpenAI(
        model=st.session_state.model,
        api_key=st.session_state.openai_api_key
    )

    # Chain: Prompt -> LLM -> Parser
    chain = system_prompt | llm | parser

    # Invoke & Render
    try:
        with st.spinner("Generating course outline..."):
            result = chain.invoke({
                "topic": st.session_state.user_input,
                "format_instructions": parser.get_format_instructions()
            })

        # Not an AI/ML topic
        if not result.is_ai_topic:
            st.warning(f"'{st.session_state.user_input}' is not an AI/ML topic. Try something like 'Transformers' or 'RAG'.")

        # Valid AI/ML topic - render course details
        else:
            st.success("Course outline generated successfully!")
            st.subheader(f"{result.course_name}")
            st.write(f"**Duration:** {result.duration}")
            st.write(f"**Difficulty:** {result.difficulty} / 5")
            st.write("**Modules:**")
            for i, module in enumerate(result.modules, start=1):
                st.write(f"  {i}. {module}")

    except Exception as e:
        st.error(f"Something went wrong: {e}")

# Validation Warnings
elif button:
    if not st.session_state.openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    elif not st.session_state.user_input:
        st.warning("Please enter a topic to generate a course.")
