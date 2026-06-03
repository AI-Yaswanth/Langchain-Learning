# LLM Basics

Learn the fundamentals of Large Language Models (LLMs) through hands-on projects built with LangChain, Streamlit, OpenAI, tokenization tools, and vector databases.

This module covers prompt engineering, LLM apps, token costs, context windows, embeddings, semantic search, and RAG fundamentals.

---

## Projects Included

### 1. Beginner AI Explainer

Explains any topic in beginner-friendly, technical, and interview-ready formats using LangChain and Streamlit.

### 2. Token Cost Calculator

Calculates token count and estimates input cost across selected LLMs.

### 3. Context Analyzer

Analyzes input and output token usage against a model’s context window to show remaining available context.

### 4. Semantic Search Engine

Uploads PDFs, creates embeddings, stores them in FAISS, and answers questions using retrieved document context.

### 5. Prompt Practice Exercises

Reusable prompts for summarization, action item extraction, Python code generation, and error explanation.

---

## Features

* Beginner-friendly AI explanations
* Prompt templates with LangChain
* Token counting with `tiktoken`
* LLM input cost estimation
* Context window usage analysis
* PDF loading and text chunking
* Embedding generation with OpenAI embeddings
* FAISS-based semantic search
* Basic RAG workflow
* Prompt engineering practice examples

---

## Tech Stack

| Layer         | Technology           |
| ------------- | -------------------- |
| LLM Framework | LangChain            |
| Models        | GPT-4o / GPT-4o-mini |
| Embeddings    | OpenAI Embeddings    |
| Vector Store  | FAISS                |
| Tokenization  | tiktoken             |
| Frontend      | Streamlit            |
| Language      | Python 3.9+          |

---

## Project Structure

```text
LLM Basics/
├── Beginner AI Explainer/
│   └── app.py
├── Token Cost Calculator/
│   └── app.py
├── Context Analyzer/
│   └── app.py
├── Semantic Search Engine/
│   └── app.py
└── LLM Prompt Exercises.md
```

---

## Files

### `Beginner AI Explainer/app.py`

Streamlit app that uses `ChatPromptTemplate` and `ChatOpenAI` to generate structured explanations.

### `Token Cost Calculator/app.py`

Counts tokens with `tiktoken` and estimates input cost based on selected model pricing.

### `Context Analyzer/app.py`

Sends a user prompt to an OpenAI model, counts input and output tokens, compares total usage against the model context limit, and displays remaining context.

### `Semantic Search Engine/app.py`

Loads a PDF, splits it into chunks, creates embeddings, stores them in FAISS, retrieves relevant chunks, and generates answers using a RAG-style workflow.

### `LLM Prompt Exercises.md`

Contains prompt practice examples for summarization, extraction, code generation, and debugging.

---

## How It Works

### Beginner AI Explainer

```text
User Query → Prompt Template → LLM → Structured Explanation
```

### Token Cost Calculator

```text
User Text → Tokenizer → Token Count → Pricing Formula → Estimated Cost
```

### Context Analyzer

```text
User Prompt → LLM Response → Input Tokens + Output Tokens → Context Usage Report
```

### Semantic Search Engine

```text
PDF → Loader → Text Splitter → Embeddings → FAISS → Retriever → LLM Answer
```

---

## Getting Started

### Prerequisites

* Python 3.9+
* OpenAI API key

### Installation

```bash
pip install streamlit langchain-openai langchain-core langchain-community langchain-text-splitters faiss-cpu pypdf tiktoken
```

---

## Running the Projects

### Beginner AI Explainer

```bash
cd "Beginner AI Explainer"
streamlit run app.py
```

### Token Cost Calculator

```bash
cd "Token Cost Calculator"
streamlit run app.py
```

### Context Analyzer

```bash
cd "Context Analyzer"
streamlit run app.py
```

### Semantic Search Engine

```bash
cd "Semantic Search Engine"
streamlit run app.py
```

Open the Streamlit app at:

```text
http://localhost:8501
```

---

## Prompt Practice

The prompt exercises demonstrate:

* Role-based prompting
* Structured output formatting
* Constraint-based prompting
* Task-specific prompt design

Example:

```text
You are a senior Python software engineer.
Generate clean, maintainable, and production-ready Python code.
```

---

## Learning Outcomes

After completing this module, you will understand:

* LLM basics and prompt engineering
* LangChain prompt templates
* LCEL chaining
* Token counting and pricing
* Context window limits
* Input vs output token usage
* Embeddings and vector stores
* Semantic search
* Retrieval-Augmented Generation basics
* Building simple AI apps with Streamlit

---

## License

MIT
