# import required libraries
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

# Page header
st.title("Session-Aware Conversational Chatbot")
st.write("A general-purpose chatbot that remembers conversation history per session.")

# Sidebar: API key, model, and session selection
st.sidebar.title("Configuration")

api_key = st.sidebar.text_input("API Key", type="password")
model = st.sidebar.selectbox("Model", ["gpt-4o", "gpt-4o-mini"])
session_id = st.sidebar.selectbox("Select your session ID", ["Session 1", "Session 2", "Session 3"])

# Session store kept in st.session_state so it persists across Streamlit reruns
# Each session_id maps to its own InMemoryChatMessageHistory instance

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Returns existing history for the session, or creates a new one."""
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]

# Prompt template
# MessagesPlaceholder injects the full chat history into each request

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a smart, friendly, and helpful conversational AI assistant.

        ### Behavior Rules:
        1. Maintain context across the conversation - remember what the user has said earlier in the session and refer back to it when relevant.
        2. Be concise but thorough - avoid unnecessary filler, but never sacrifice clarity for brevity.
        3. If the user asks a follow-up question, treat it in the context of the ongoing conversation, not as a standalone query.
        4. If you don't know something or are unsure, say so honestly - do not fabricate information.
        5. Match the user's tone - be formal when they are formal, casual when they are casual.
        6. If the user's message is vague or unclear, ask one clarifying question before proceeding.

        ### Boundaries:
        - You do not have access to any external systems, databases, or knowledge bases.
        - You cannot look up real-time information (e.g. live prices, news, order status).
        - If asked for something outside your knowledge, clearly say so and suggest where they might find it.

        ### Response Format:
        - Use bullet points or numbered lists only when presenting multiple items.
        - Use plain prose for conversational replies.
        - Keep responses focused - avoid padding or restating the question back.
        """ 
        ),
    MessagesPlaceholder(
        "history"
        ),
    (
        "human",
        "{input}"
        )
    ])

# LLM setup
llm = ChatOpenAI(model=model, api_key=api_key)

# Main input

user_query = st.text_input("Enter your query")
btn_proceed = st.button("Proceed")

# Core logic: runs only when all inputs are provided and button is clicked
if api_key and session_id and user_query and btn_proceed:

    # Build the chain: prompt -> LLM
    chain = prompt | llm

    # Wrap chain with session-aware history manager.
    # Pass the function reference (not a call) so LangChain invokes it internally.
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        history_messages_key="history",
        input_messages_key="input"
    )

    with st.spinner("Thinking..."):
        try:
            # Pass session_id via config so LangChain routes to the correct history
            response = chain_with_history.invoke(
                {"input": user_query},
                config={"configurable": {"session_id": session_id}}
            )

            st.success("Response:")
            st.write(response.content)

        except Exception as e:
            st.error(f"Error: {e}")

# Validation messages only shown after button click to avoid noise on load
elif btn_proceed:
    if not api_key:
        st.info("Please enter your **OpenAI API key** to get started.")
    elif not user_query:
        st.info("Please enter a **query** to get started.")
