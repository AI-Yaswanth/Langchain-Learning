import streamlit as st
import tiktoken

# Tokenizer
encoding = tiktoken.get_encoding("cl100k_base")

st.title("Token Cost Calculator")

# Model selection
model = st.sidebar.selectbox(
    "Select Model",
    ["gpt-4o", "gpt-4o-mini", "Claude Sonnet 4.5", "Claude Haiku 4.5"]
)

# User input
user_input = st.text_area(
    "Enter your text here",
    height=200
)

# Pricing (input cost per million tokens)
PRICING = {
    "gpt-4o": 2.50,
    "gpt-4o-mini": 0.15,
    "Claude Sonnet 4.5": 3.00,
    "Claude Haiku 4.5": 1.00
}

calculate_btn = st.button("Calculate")

if user_input and calculate_btn:
    tokens = encoding.encode(user_input)
    token_count = len(tokens)

    cost_per_million = PRICING[model]
    estimated_cost = (token_count / 1_000_000) * cost_per_million

    st.subheader("Results")
    st.write(f"**Model:** {model}")
    st.write(f"**Token Count:** {token_count:,}")
    st.write(f"**Estimated Input Cost:** ${estimated_cost:.8f}")

    if "Claude" in model:
        st.info(
            "Claude token counts are approximated using OpenAI's "
            "cl100k_base tokenizer and may differ slightly."
        )

elif not user_input:
    st.error("Please enter some text")
