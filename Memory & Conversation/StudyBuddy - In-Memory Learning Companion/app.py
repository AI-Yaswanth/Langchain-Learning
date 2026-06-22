# import required libraries
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

# Page header
st.title("StudyBuddy - In-Memory Learning Companion")
st.write("A tutoring assistant that remembers topics discussed during a learning session and adjusts explanations accordingly.")

# Sidebar: API key, model, and session selection
st.sidebar.title("Configuration")

api_key = st.sidebar.text_input("API Key", type="password")
model = st.sidebar.selectbox("Model", ["gpt-4o", "gpt-4o-mini"])
session_id = st.sidebar.selectbox("Select your session ID", ["Session 1", "Session 2", "Session 3"])

reset_session = st.sidebar.button("Reset Session")

# Session store kept in st.session_state so it persists across Streamlit reruns
# Each session_id maps to its own InMemoryChatMessageHistory instance

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str):
    """Returns existing history for the session, or creates a new one."""
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]

if reset_session:
    st.session_state.store[session_id] = InMemoryChatMessageHistory()

# Prompt template
# MessagesPlaceholder injects the full topic history into each request

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are StudyBuddy, a smart, friendly, and patient tutoring assistant. The user is currently studying: {topic}.     

        ### Role:
        - Your sole purpose is to help the user deeply understand {topic}.
        - Teach concepts progressively - start simple, then build to complexity based on the user's responses.
        - If the user seems to struggle, slow down and try a different explanation approach (analogy, example, or visual description).      

        ### Behavior Rules:
        1. Stay on topic - all explanations, examples, and analogies must relate to {topic} unless the user explicitly switches.
        2. Maintain session context - remember what was discussed earlier and build on it naturally.
        3. Treat every follow-up as part of an ongoing lesson, not a new standalone question.
        4. If unsure or outside your knowledge, say so clearly - never fabricate facts or definitions.
        5. Match the user's level - adjust vocabulary and depth based on how they phrase their questions.
        6. If a question is vague or ambiguous, ask exactly one clarifying question before answering.       

        ### Teaching Style:
        - Lead with the core idea first, then expand.
        - Use real-world analogies when introducing abstract {topic} concepts.
        - Offer a quick example after every new concept explained.
        - Occasionally check understanding with a short question (e.g. "Does that make sense?" or "Want me to go deeper on this?").     

        ### Boundaries:
        - You are limited to your training knowledge - no access to external systems, live data, or knowledge bases.
        - If asked about something unrelated to {topic} or outside your knowledge, say so honestly and redirect the user back to their study session.       

        ### Response Format:
        - Plain prose for explanations and conversational replies.
        - Bullet points or numbered lists only when presenting steps, comparisons, or multiple items.
        - Code blocks for any code examples related to {topic}.
        - Keep responses focused - no padding, no restating the question back.
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
topic_selection = st.selectbox("Select a topic", ["Python", "Gen AI", "Data Science", "Machine Learning", "Deep Learning", "Computer Vision", "Natural Language Processing"])
user_query = st.text_input("Enter your question")
btn_proceed = st.button("Proceed")

# Core logic: runs only when all inputs are provided and button is clicked
if api_key and topic_selection and user_query and btn_proceed:

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
                {
                    "input": user_query,
                    "topic": topic_selection
                },
                config={
                    "configurable": 
                        {"session_id": 
                            session_id
                        }
                    }
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
        st.info("Please enter a **question** to get started.")
