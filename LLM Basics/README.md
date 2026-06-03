# LLM Basics

Learn the fundamentals of Large Language Models (LLMs) through hands-on projects built with LangChain, Streamlit, OpenAI, and vector databases. This module covers prompt engineering, LLM-powered applications, tokenization, embeddings, and semantic search - the building blocks of modern AI applications.

---

## Projects Included

### 1. Beginner AI Explainer

A lightweight web app that explains any topic in multiple formats - beginner-friendly, technical, and interview-ready. Demonstrates prompt templates and LLM integration using LangChain.

### 2. Token Cost Calculator

A utility application that estimates token usage and input costs for different LLMs. Demonstrates tokenization concepts and pricing awareness when building AI applications.

### 3. Semantic Search Engine

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDFs, create embeddings, perform semantic search, and generate context-aware answers using retrieved document chunks.

### 4. Prompt Practice Exercises

A collection of practical prompts designed to improve prompt engineering skills across common real-world use cases.

---

## Features

### Beginner AI Explainer

* Structured AI-generated explanations
* Beginner and technical learning modes
* Interview-focused answers
* LangChain Prompt Templates
* OpenAI model integration

### Token Cost Calculator

* Token counting using `tiktoken`
* Multi-model cost estimation
* Pricing comparison across models
* Cost awareness for production AI systems

### Semantic Search Engine

* PDF document upload
* Automatic document chunking
* Vector embeddings generation
* FAISS vector database integration
* Semantic document retrieval
* Context-aware question answering (RAG)

### Prompt Practice Exercises

* Document summarization prompts
* Action item extraction prompts
* Python code generation prompts
* Error explanation prompts

---

## Tech Stack

| Layer           | Technology           |
| --------------- | -------------------- |
| LLM Framework   | LangChain            |
| Embeddings      | OpenAI Embeddings    |
| Vector Store    | FAISS                |
| Tokenization    | tiktoken             |
| Language Models | GPT-4o / GPT-4o-mini |
| Frontend        | Streamlit            |
| Language        | Python 3.9+          |

---

## Project Structure

```text
LLM Basics/
├── Beginner AI Explainer/
│   └── app.py
├── Token Cost Calculator/
│   └── app.py
├── Semantic Search Engine/
│   └── app.py
└── LLM Prompt Exercises.md
```

---

## Files

### `Beginner AI Explainer/app.py`

Demonstrates:

* Prompt engineering with `ChatPromptTemplate`
* LangChain Expression Language (LCEL)
* OpenAI chat models
* Streamlit-based UI development

### `Token Cost Calculator/app.py`

Demonstrates:

* Tokenization using `tiktoken`
* Token counting
* LLM pricing calculations
* Cost estimation workflows

### `Semantic Search Engine/app.py`

Demonstrates:

* PDF document ingestion
* Text chunking with `RecursiveCharacterTextSplitter`
* Embedding generation using `OpenAIEmbeddings`
* Vector storage using FAISS
* Semantic retrieval
* Retrieval-Augmented Generation (RAG)
* Context-aware question answering

### `LLM Prompt Exercises.md`

Contains reusable prompts for:

| Prompt Type           | Purpose                           |
| --------------------- | --------------------------------- |
| Document Summarizer   | Summarize documents accurately    |
| Action Item Extractor | Extract tasks from meetings       |
| Python Code Generator | Generate production-ready code    |
| Error Explainer       | Debug and explain software errors |

---

## How It Works

### Beginner AI Explainer

```text
User Query
     │
     ▼
Prompt Template
     │
     ▼
OpenAI Model
     │
     ▼
Structured Explanation
```

### Token Cost Calculator

```text
User Text
     │
     ▼
Tokenizer
     │
     ▼
Token Count
     │
     ▼
Pricing Model
     │
     ▼
Estimated Cost
```

### Semantic Search Engine (RAG Workflow)

```text
PDF Upload
     │
     ▼
Document Loader
     │
     ▼
Text Splitter
     │
     ▼
Embeddings
     │
     ▼
FAISS Vector Store
     │
     ▼
Retriever
     │
     ▼
Relevant Chunks
     │
     ▼
LLM
     │
     ▼
Answer
```

Example retrieval flow:

```python
retrieved_docs = retriever.invoke(user_query)

response = chain.invoke({
    "context": retrieved_docs,
    "input": user_query
})
```

---

## Getting Started

### Prerequisites

* Python 3.9+
* OpenAI API Key

### Installation

```bash
pip install streamlit \
            langchain-openai \
            langchain-core \
            langchain-community \
            langchain-text-splitters \
            faiss-cpu \
            pypdf \
            tiktoken
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

### Semantic Search Engine

```bash
cd "Semantic Search Engine"
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## Prompt Practice

The prompt exercises demonstrate key prompt engineering techniques:

### Role Prompting

```text
You are a senior Python software engineer.
```

### Structured Output Prompting

* Tables
* Bullet points
* Multi-section responses

### Constraint-Based Prompting

```text
Do not invent information.
Use only the provided context.
```

### Task-Specific Prompt Design

* Summarization
* Information extraction
* Code generation
* Debugging assistance

---

## Learning Outcomes

After completing this module, you will understand:

### LLM Fundamentals

* Prompt engineering
* Prompt templates
* Chat-based LLM interactions

### LangChain Fundamentals

* LCEL chains
* Prompt templates
* Output parsers

### Embeddings & Retrieval

* Vector embeddings
* Similarity search
* FAISS vector databases
* Document chunking

### RAG Fundamentals

* Retrieval-Augmented Generation
* Context injection
* Grounded responses
* Semantic search systems

### Production Considerations

* Token counting
* Cost estimation
* Model selection
* API usage optimization

---

## License

MIT
