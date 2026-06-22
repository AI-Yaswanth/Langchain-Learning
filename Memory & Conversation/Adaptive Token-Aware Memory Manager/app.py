import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import SystemMessage
import tiktoken

# Compression triggers when token count exceeds TOKEN_LIMIT;
# only the most recent KEEP_RECENT_MESSAGES messages are preserved verbatim.
TOKEN_LIMIT = 2000
KEEP_RECENT_MESSAGES = 6

if "response" not in st.session_state:
    st.session_state.response = None


def count_tokens(messages: list) -> int:
    """Returns total token count across all messages using cl100k_base encoding."""
    encoder = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoder.encode(msg.content)) for msg in messages)


def compress_history(history, llm) -> bool:
    """
    Summarizes older messages via the LLM and rebuilds history as:
    [summary system message] + [KEEP_RECENT_MESSAGES most recent messages].
    Returns False if history is too short to compress.
    """
    if len(history.messages) <= KEEP_RECENT_MESSAGES:
        return False

    old_messages = history.messages[:-KEEP_RECENT_MESSAGES]
    recent_messages = history.messages[-KEEP_RECENT_MESSAGES:]

    summary_prompt = f"""
        Summarize this conversation.

        Focus on:
            - User goals
            - Important facts
            - Decisions made
            - Context needed for future replies

        Conversation: {old_messages}

        Summary:
        """

    summary = llm.invoke(summary_prompt).content

    history.clear()
    history.add_message(
        SystemMessage(
            content=f"""
                Conversation Summary: {summary}
                Use this summary as memory for future responses.
                """
        )
    )
    for msg in recent_messages:
        history.add_message(msg)

    return True


def manage_memory(history, llm) -> bool:
    """Compresses history if token usage exceeds TOKEN_LIMIT."""
    if count_tokens(history.messages) > TOKEN_LIMIT:
        return compress_history(history, llm)
    return False


# Persisted across Streamlit reruns via session_state; keyed by session_id.
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Returns the chat history for the given session, creating it if absent."""
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]


# UI

st.title("Adaptive Token-Aware Memory Manager")
st.write("A memory manager that remembers topics discussed during a session and compresses history when the token limit is reached.")

st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("API Key", type="password")
model = st.sidebar.selectbox("Model", ["gpt-4o", "gpt-4o-mini"])
session_id = st.sidebar.selectbox("Select your session ID", ["Session 1", "Session 2"])

st.sidebar.markdown("---")
history = get_session_history(session_id)

msg_metric = st.sidebar.empty()
token_metric = st.sidebar.empty()

def update_metrics():
    msg_metric.metric("Total Messages", len(history.messages))
    token_metric.metric("Total Tokens", count_tokens(history.messages))

update_metrics()

if st.sidebar.button("Clear Memory"):
    history.clear()
    st.success("Memory cleared.")
    st.rerun()


# MessagesPlaceholder injects the full conversation history on every invocation.
# If compression has run, the history begins with a summary system message.
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful assistant.

        If a conversation summary exists in the system message,
        use it to remember older context.

        Keep responses concise.
        """
    ),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

llm = ChatOpenAI(model=model, api_key=api_key)

user_query = st.text_input("Enter your question")
btn_proceed = st.button("Proceed")

if api_key and user_query and btn_proceed:
    chain = prompt | llm

    # get_session_history is passed by reference; LangChain calls it with session_id internally.
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        history_messages_key="history",
        input_messages_key="input"
    )

    with st.spinner("Thinking..."):
        try:
            response = chain_with_history.invoke(
                {"input": user_query},
                config={"configurable": {"session_id": session_id}}
            )

            st.session_state.response = response.content

            if st.session_state.response:
                st.success("Response:")
                st.write(st.session_state.response)
                update_metrics()

            # Compression runs after the response so the new exchange is included.
            compressed = manage_memory(history, llm)
            if compressed:
                st.success("Memory compressed and summarized.")

        except Exception as e:
            st.error(f"Error: {e}")

elif btn_proceed:
    if not api_key:
        st.info("Please enter your **OpenAI API key** to get started.")
    elif not user_query:
        st.info("Please enter a **question** to get started.")
