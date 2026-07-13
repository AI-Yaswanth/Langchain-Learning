# RAG (Retrieval-Augmented Generation)

Learn how to build Retrieval-Augmented Generation (RAG) applications using **LangChain**, **OpenAI**, **ChromaDB**, and **Streamlit**.

This module covers document loading, text chunking, embedding generation, vector databases, semantic search, retrieval strategies, metadata filtering, enterprise knowledge retrieval, and grounded AI responses.

---

## Projects Included

### 1. Multi-PDF Intelligent Q&A Assistant

Upload multiple PDF documents, generate embeddings, and ask natural language questions that are answered strictly using the uploaded documents.

### 2. Enterprise Knowledge Base Chatbot

Build an enterprise-ready document chatbot with role-based document access, metadata tagging, secure retrieval, and source-aware responses.

### 3. Research Paper Semantic Search Engine

Perform semantic search across research papers using embeddings, similarity scoring, metadata filtering, and preview snippets.

### 4. Retrieval Strategy Comparator

Compare Similarity Search, Max Marginal Relevance (MMR), and Similarity Score Threshold retrieval strategies to understand how different retrieval methods affect search quality.

### 5. I Don't Know RAG Bot

Build a hallucination-resistant RAG chatbot that answers only when sufficient document evidence exists and explicitly responds with **"I don't know"** when relevant information cannot be retrieved.

---

## Features

* Retrieval-Augmented Generation (RAG)
* Multi-Document Question Answering
* PDF, CSV, and Text Document Loading
* Recursive Text Chunking
* OpenAI Embedding Generation
* Chroma Vector Database
* Semantic Search
* Similarity Search
* Max Marginal Relevance (MMR)
* Similarity Score Threshold Retrieval
* Metadata-Based Retrieval
* Role-Based Document Access
* Source-Aware Responses
* Hallucination Mitigation
* Grounded AI Responses
* Streamlit-Based User Interfaces

---

## Tech Stack

| Layer            | Technology                                   |
| ---------------- | -------------------------------------------- |
| Framework        | LangChain                                    |
| Models           | GPT-4o / GPT-4o-mini                         |
| Embeddings       | OpenAI Embeddings (`text-embedding-3-small`) |
| Vector Database  | ChromaDB                                     |
| Document Loaders | PyPDFLoader, CSVLoader, TextLoader           |
| Text Splitting   | RecursiveCharacterTextSplitter               |
| Frontend         | Streamlit                                    |
| Environment      | python-dotenv                                |
| Language         | Python 3.9+                                  |

---

## Project Structure

```text
RAG/
├── Multi-PDF Intelligent Q&A Assistant/
│   └── app.py
├── Enterprise Knowledge Base Chatbot/
│   └── app.py
├── Research Paper Semantic Search Engine/
│   ├── app.py
│   └── .env.example
├── Retrieval Strategy Comparator/
│   └── app.py
└── I Don't Know RAG Bot/
    └── app.py
```

---

## Files

### `Multi-PDF Intelligent Q&A Assistant/app.py`

Implements a complete RAG pipeline that loads multiple PDFs, generates embeddings, stores them in ChromaDB, retrieves relevant context, and produces grounded answers using an LLM.

### `Enterprise Knowledge Base Chatbot/app.py`

Builds an enterprise knowledge assistant with metadata tagging, role-based document access, filtered retrieval, and source-aware responses.

### `Research Paper Semantic Search Engine/app.py`

Implements semantic search across research papers with metadata filtering, similarity scoring, and preview snippets for retrieved documents.

### `Retrieval Strategy Comparator/app.py`

Demonstrates and compares Similarity Search, Max Marginal Relevance (MMR), and Similarity Score Threshold retrieval strategies side-by-side.

### `I Don't Know RAG Bot/app.py`

Builds a grounded RAG chatbot that answers questions only when sufficient supporting context is retrieved, helping reduce hallucinations.

### `.env.example`

Template for configuring the OpenAI API key required by the Research Paper Semantic Search Engine.

---

## How It Works

### Multi-PDF Intelligent Q&A Assistant

```text
PDFs
   ↓
Document Loader
   ↓
Text Splitter
   ↓
Embeddings
   ↓
ChromaDB
   ↓
Retriever
   ↓
LLM
   ↓
Grounded Answer
```

### Enterprise Knowledge Base Chatbot

```text
Documents
      ↓
Metadata Tagging
      ↓
Embeddings
      ↓
ChromaDB
      ↓
Role-Based Retriever
      ↓
LLM
      ↓
Source-Aware Response
```

### Research Paper Semantic Search Engine

```text
Research Papers
        ↓
Document Loader
        ↓
Embeddings
        ↓
ChromaDB
        ↓
Similarity Search
        ↓
Metadata Filtering
        ↓
Ranked Results
```

### Retrieval Strategy Comparator

```text
Documents
      ↓
Embeddings
      ↓
ChromaDB
      ↓
Similarity Search
MMR Search
Threshold Search
      ↓
Result Comparison
```

### I Don't Know RAG Bot

```text
User Question
      ↓
Retriever
      ↓
Relevant Context
      ↓
Grounded Prompt
      ↓
LLM
      ↓
Answer or "I Don't Know"
```

---

## Getting Started

### Prerequisites

* Python 3.9+
* OpenAI API Key

### Installation

```bash
pip install streamlit langchain langchain-core langchain-classic langchain-community langchain-openai chromadb pypdf python-dotenv
```

---

## Running the Projects

### Multi-PDF Intelligent Q&A Assistant

```bash
cd "Multi-PDF Intelligent Q&A Assistant"
streamlit run app.py
```

### Enterprise Knowledge Base Chatbot

```bash
cd "Enterprise Knowledge Base Chatbot"
streamlit run app.py
```

### Research Paper Semantic Search Engine

```bash
cd "Research Paper Semantic Search Engine"
streamlit run app.py
```

### Retrieval Strategy Comparator

```bash
cd "Retrieval Strategy Comparator"
streamlit run app.py
```

### I Don't Know RAG Bot

```bash
cd "I Don't Know RAG Bot"
streamlit run app.py
```

Open the application at:

```text
http://localhost:8501
```

---

## Learning Outcomes

After completing this module, you will understand:

* Retrieval-Augmented Generation (RAG)
* Document Loading with LangChain
* Recursive Text Chunking
* Embedding Generation with OpenAI
* Vector Databases using ChromaDB
* Semantic Search
* Similarity Search
* Max Marginal Relevance (MMR)
* Similarity Score Threshold Retrieval
* Metadata-Based Retrieval
* Role-Based Document Access
* Retrieval Chains
* Grounded Question Answering
* Hallucination Mitigation
* Building Production-Ready RAG Applications

---

## License

MIT
