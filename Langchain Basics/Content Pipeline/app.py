# Import necessary libraries
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


st.title("Content Pipeline")
st.write("Create a content pipeline for a blog post.")

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
    label="Topic"
)
button = st.button("Create Content Pipeline")

# Run parallel analysis when button is clicked
if button and st.session_state.openai_api_key and st.session_state.user_input:
    
    llm = ChatOpenAI(
        api_key=st.session_state.openai_api_key,
        model=st.session_state.model
    )
    
    blog_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert content writer. Write an engaging, well-structured blog post on the topic: {topic}. "
            "The post must include a compelling headline, an introduction, 2-3 main sections with subheadings, "
            "and a conclusion. Keep it within 500 words. Use a conversational yet professional tone."
        )
    ])
    
    linkedin_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a professional LinkedIn content creator. Write an impactful LinkedIn post on the topic: {topic}. "
            "Start with a strong hook, share a key insight or takeaway, and end with a call-to-action or thought-provoking question. "
            "Keep it within 150 words. Use short paragraphs and relevant emojis sparingly."
        )
    ])
    
    tweet_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a social media expert. Write a punchy, engaging tweet on the topic: {topic}. "
            "Make it concise, use plain language, and include 1-2 relevant hashtags. "
            "Stay strictly within 280 characters."
        )
    ])
    
    summary_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a concise and accurate summarizer. Summarize the key insights on the topic: {topic} "
            "in exactly 5 bullet points. Each bullet should be clear, actionable, and no longer than one sentence. "
            "Use '•' as the bullet symbol."
        )
    ])
    
    blog_chain = blog_prompt | llm | StrOutputParser()
    linkedin_chain = linkedin_prompt | llm | StrOutputParser()
    tweet_chain = tweet_prompt | llm | StrOutputParser()
    summary_chain = summary_prompt | llm | StrOutputParser()
    
    mapping_chain = RunnablePassthrough() | {
        "blog": blog_chain,
        "linkedin": linkedin_chain,
        "tweet": tweet_chain,
        "summary": summary_chain
    }
    with st.spinner("Creating content pipeline..."):
        content_pipeline = mapping_chain.invoke({"topic": st.session_state.user_input})

        st.markdown("## Blog")
        st.write(content_pipeline.get("blog"))
        st.divider()
        st.markdown("## LinkedIn Post")
        st.write(content_pipeline.get("linkedin"))
        st.divider()
        st.markdown("## Tweet")
        st.write(content_pipeline.get("tweet"))
        st.divider()
        st.markdown("## Summary")
        st.write(content_pipeline.get("summary"))
    
# Show warnings if inputs are missing
elif button:
    if not st.session_state.openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    elif not st.session_state.user_input:
        st.warning("Please enter a content text to create a content pipeline.")
