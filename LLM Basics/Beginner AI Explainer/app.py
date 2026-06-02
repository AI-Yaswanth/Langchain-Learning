# Import necessary libraries
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

# Set the main title of the app
st.title("Beginner AI Explainer")

# Create a settings panel in the sidebar
st.sidebar.title("Settings")

# Get API key from user (password field hides input)
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Let user choose which OpenAI model to use
model = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])

# Define the prompt template with system instructions and user input format
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            # System message tells the AI how to behave
            """You are a knowledgeable, beginner-friendly, and technically accurate assistant. Your task is to explain any user-requested topic in a clear, structured, and easy-to-understand manner.

            ## Instructions
            - Start from basic concepts and progressively move to advanced concepts.
            - Keep explanations accurate, concise, and engaging.
            - Use Markdown formatting for readability.
            - Use tables where appropriate.
            - If the topic is complex, break it into smaller sections.
            - Prefer practical examples over theoretical explanations.
            - Tailor the depth of explanation based on the topic's complexity.
            
            ## Response Format
            
            ### Topic
            Provide a one-line definition of the topic.
            
            ## 1. Simple Explanation (Beginner-Friendly)
            - Explain the concept using simple, everyday language.
            - Avoid technical jargon whenever possible.
            - Use relatable examples or analogies.
            - Assume the reader has no prior knowledge of the topic.
            
            ### Example
            Provide a simple real-world example that demonstrates the concept.
            
            ## 2. Technical Explanation
            - Explain the concept using proper technical terminology.
            - Cover important components, architecture, workflow, principles, or mechanisms.
            - Include relevant details that someone with a technical background would expect.
            - Use bullet points and diagrams (ASCII/Markdown) when helpful.
            
            ### Technical Example
            Provide a practical technical example or implementation scenario.
            
            ## 3. Interview Answer
            Provide a concise and professional interview-ready answer that:
            - Starts with a clear definition.
            - Covers key concepts and benefits.
            - Highlights practical use cases.
            - Can be delivered confidently within 1–2 minutes.
            
            ### Sample Interview Answer
            Write the answer as if the candidate is responding directly to an interviewer.
            
            ## Real-World Use Cases
            List common real-world applications of the topic.
            
            ## Advantages
            - Advantage 1
            - Advantage 2
            - Advantage 3
            
            ## Disadvantages
            - Disadvantage 1
            - Disadvantage 2
            - Disadvantage 3
            
            ## Common Interview Follow-up Questions
            ### Q1: <Question>
            **Answer:** <Short answer>
            
            ### Q2: <Question>
            **Answer:** <Short answer>
            
            ### Q3: <Question>
            **Answer:** <Short answer>"""
        ),
        (
            "user",
            # User message template - {user_query} will be replaced with actual user input
            "user query: {user_query}"
        )
    ]
)

# Initialize the ChatOpenAI model with selected settings
llm = ChatOpenAI(
    model=model,
    openai_api_key=api_key,
)

# Create a text input field for users to enter their query
st.session_state.user_query = st.text_input(
    "Enter your query"
    )

# Create an "Explain" button for the user to click
btn_explain = st.button("Explain")

# Check if user has provided all required inputs
if api_key and st.session_state.user_query and btn_explain:
    # Create a chain by connecting the prompt and language model
    chain = prompt | llm
    
    # Send the user query through the chain and get the response
    response = chain.invoke({
        "user_query": st.session_state.user_query
    })
    
    # Display the AI's response
    st.write(response.content)

# Show helpful messages if required inputs are missing
elif not api_key:
    st.info("Please enter your OpenAI API key to get started.")
elif not st.session_state.user_query:
    st.info("Please enter a query to get started.")
elif not btn_explain:
    st.info("Please click the Explain button to get started.")
