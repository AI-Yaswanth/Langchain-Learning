# Importing the necessary libraries
import streamlit as st
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Initialize tokenizer for token counting (using GPT-4/3.5 encoding)
encoding = tiktoken.get_encoding("cl100k_base")

# Model context size limits (tokens)
CONTEXT_SIZES = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000
}

# Page configuration
st.title("Token Context Size Calculator")

# SIDEBAR: USER INPUT & CONFIGURATION

st.sidebar.header("Settings")

# Get OpenAI API key from user
api_key = st.sidebar.text_input(
    "Enter your OpenAI API key",
    type="password",
    help="Your API key is not stored or logged"
)

# Model selection dropdown
selected_model = st.sidebar.selectbox(
    "Select Model",
    options=list(CONTEXT_SIZES.keys()),
    help="Choose the OpenAI model to analyze token usage for"
)

# Text area for user input
user_input = st.text_input(
    "Enter your text or prompt here",
    placeholder="Paste your text content..."
)

# Calculate button
calculate_btn = st.button("Calculate Tokens")

# PROCESSING & RESULTS
if calculate_btn:
    # Validate inputs
    if not user_input:
        st.error("Please enter some text to calculate tokens")
        st.stop()
    
    if not api_key:
        st.error("Please enter your OpenAI API key")
        st.stop()
    
    try:
        # Initialize LLM and create prompt chain
        llm = ChatOpenAI(api_key=api_key, model=selected_model)
        prompt_template = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Answer the following question:\n\n{question}"
        )
        chain = prompt_template | llm
        
        # Show processing status
        with st.spinner("Processing your request..."):
            # Generate response from the model
            response = chain.invoke({"question": user_input})
        
        # TOKEN COUNTING
        
        # Count tokens in the original user input
        input_tokens = len(encoding.encode(user_input))
        
        # Count tokens in the model's response
        output_tokens = len(encoding.encode(response.content))
        
        # Calculate total tokens used
        total_tokens = input_tokens + output_tokens
        
        # Calculate remaining context available
        remaining_context = CONTEXT_SIZES[selected_model] - total_tokens
        
        # Context limit validation
        if total_tokens > CONTEXT_SIZES[selected_model]:
            st.error(
                f"**Over Limit!** Total tokens ({total_tokens:,}) exceed "
                f"the context size ({CONTEXT_SIZES[selected_model]:,}) for {selected_model}"
            )
        else:
            st.success(
                f"**Within Limit!** Total tokens ({total_tokens:,}) are within "
                f"the context size ({CONTEXT_SIZES[selected_model]:,}) for {selected_model}"
            )
            
        # Key metrics display
        st.write("**Total Tokens:**", f"{total_tokens:,}")
        st.write("**Context Usage:**", f"{(total_tokens / CONTEXT_SIZES[selected_model] * 100):.1f}%")
        st.write("**Remaining Context:**", f"{remaining_context:,}")
        
        # Detailed breakdown
        st.write(f"**Model:** {selected_model}")
        st.write(f"**Input Tokens:** {input_tokens:,}")
        st.write(f"**Output Tokens:** {output_tokens:,}")
        st.write(f"**Context Limit:** {CONTEXT_SIZES[selected_model]:,}")
        st.write(f"**Total Tokens Used:** {total_tokens:,}")
        st.write(f"**Remaining Context:** {remaining_context:,}")
    
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        st.info("Please check your API key and try again")
