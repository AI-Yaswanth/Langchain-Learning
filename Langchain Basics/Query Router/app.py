from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from typing import Literal
from pydantic import BaseModel
from langchain_core.runnables import RunnableLambda

# Page Config
st.set_page_config(page_title="Query Router", layout="centered")

st.title("Query Router")
st.write(
    "Enter a query to classify it into one of the following categories: "
    "**Coding**, **Technical**, or **General** - and get an appropriate response."
)

# Sidebar Settings
st.sidebar.title("Settings")

openai_api_key = st.sidebar.text_input(
    label="OpenAI API Key:",
    type="password",
    placeholder="sk-..."
)

model = st.sidebar.selectbox(
    label="Select Model:",
    options=["gpt-4o", "gpt-4o-mini"]
)

# User Input
user_input = st.text_input(
    label="Enter a query:",
    placeholder="e.g. Write a Python function to sort a list"
)
button = st.button("Classify & Answer")

# Main Logic
if button and openai_api_key and user_input:

    # Classifier Prompt
    classifier_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an expert query classifier with 10+ years of experience in natural language understanding.
            Your task is to analyze the given user query and classify it into exactly one of the following categories:
                - coding : User wants code written, generated, debugged, or reviewed. Examples: "Write a Python function to sort a list", "Fix this code", "Generate a SQL query for..."
                - technical : User wants to understand or learn about a technology, language, tool, or concept - but is NOT asking for code. Examples: "What is Python?", "How does REST API work?", "What is the difference between SQL and NoSQL?"
                - general : Conversational, everyday, or unrelated to technology. Examples: "What's the weather today?", "Tell me a joke", "Who is Elon Musk?"

            Classification Rule:
                - User wants code produced -> coding
                - User wants knowledge/explanation -> technical
                - Unrelated to tech -> general

            Instructions:
                - Read the user query carefully.
                - Classify into one category only.
                - Respond ONLY with valid JSON. No extra text, no markdown fences.
                - JSON format: {{"category": "<coding|technical|general>", "user_query": "<original query>"}}
            """
        ),
        ("user", "{user_query}")
    ])

    # Specialist Prompts
    coding_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an expert coding assistant. When given a request:
                - Write -> Produce working code with brief comments explaining key parts
                - Debug -> Identify the root cause and show the corrected code
                - Explain -> Describe what the code does in plain, simple language
                - Review -> Give feedback on bugs, performance, and readability

            Always be concise. State any assumptions you make.
            """
        ),
        ("user", "{user_query}")
    ])

    technical_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a knowledgeable tech educator. When given a question:
                - Give a clear, plain-language explanation of the concept
                - Use analogies or real-world examples to simplify complex ideas
                - Compare or contrast technologies when relevant
                - End with a one-line takeaway if helpful

            Never write code. Keep answers focused and jargon-free.
            """
        ),
        ("user", "{user_query}")
    ])

    general_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a friendly, helpful assistant. When given a question:
                - Answer conversationally and directly
                - Keep responses short unless detail is needed
                - Be warm, natural, and easy to talk to

            Avoid unnecessary lists or formatting. Just have a helpful conversation.
            """
        ),
        ("user", "{user_query}")
    ])

    # LLM Initialisation
    llm = ChatOpenAI(model=model, api_key=openai_api_key)

    # Pydantic Output Schema
    class ClassifierOutput(BaseModel):
        category: Literal["coding", "technical", "general"]
        user_query: str

    # Individual Chains
    classifier_chain = classifier_prompt | llm | JsonOutputParser(pydantic_object=ClassifierOutput)
    coding_chain = coding_prompt | llm | StrOutputParser()
    technical_chain = technical_prompt | llm | StrOutputParser()
    general_chain = general_prompt | llm | StrOutputParser()

    # Router Function
    def perform_action(classifier_result):
        """Route to the appropriate specialist chain based on classified category."""
        category = classifier_result.get("category")
        query_dict = {"user_query": classifier_result.get("user_query")}

        if category == "coding":
            response = coding_chain.invoke(query_dict)
            
        elif category == "technical":
            response = technical_chain.invoke(query_dict)
            
        else:
            response = general_chain.invoke(query_dict)

        return {"category": category, "response": response}

    # Final Chain
    final_chain = classifier_chain | RunnableLambda(perform_action)

    try:
        with st.spinner("Classifying and generating response..."):
            result = final_chain.invoke({"user_query": user_input})

        # Display Results
        category = result.get("category", "unknown").capitalize()

        st.divider()

        # Response
        st.subheader("Response:")
        st.markdown(result.get("response", "No response generated."))

    except Exception as e:
        st.error(f"Something went wrong: {e}")

# Validation Warnings
elif button:
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    if not user_input:
        st.warning("Please enter a query to classify.")
