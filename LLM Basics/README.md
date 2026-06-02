# LLM Basics

Learn the fundamentals of working with Large Language Models (LLMs) using LangChain. This module covers prompt engineering basics and building a simple AI-powered explainer application with Streamlit.

---

## Projects Included

### 1. Beginner AI Explainer

A lightweight web app that explains any topic in multiple formats—beginner-friendly, technical, and interview-ready. Built with LangChain and Streamlit to demonstrate prompt templates and LLM integration.

### 2. Prompt Practice Exercises

A collection of practical prompts designed to strengthen prompt engineering skills for common real-world use cases such as summarization, information extraction, code generation, and debugging assistance.

---

## Features

### Beginner AI Explainer

* Structured explanations for any topic
* Beginner-friendly and technical breakdowns
* Interview-ready answers
* Model selection (`gpt-4o-mini` or `gpt-4o`)
* Streamlit-based user interface
* LangChain prompt template integration

### Prompt Practice Exercises

* Document summarization prompts
* Action item extraction prompts
* Python code generation prompts
* Error explanation and debugging prompts
* Examples of structured prompt design and output formatting

---

## Tech Stack

| Layer          | Technology                  |
| -------------- | --------------------------- |
| LLM Framework  | LangChain                   |
| Language Model | OpenAI GPT-4o / GPT-4o-mini |
| UI Framework   | Streamlit                   |
| Language       | Python 3.9+                 |

---

## Project Structure

```text
LLM Basics/
├── Beginner AI Explainer/
│   └── app.py
└── LLM Prompt Exercises.md
```

---

## Files

### `Beginner AI Explainer/app.py`

Main Streamlit application demonstrating:

* LangChain `ChatPromptTemplate`
* OpenAI model integration using `ChatOpenAI`
* User input handling
* Prompt chaining (`prompt | llm`)
* Structured AI-generated responses

### `LLM Prompt Exercises.md`

Contains prompt engineering exercises for:

| Prompt                | Purpose                                                 |
| --------------------- | ------------------------------------------------------- |
| Document Summarizer   | Generate concise summaries from documents               |
| Action Item Extractor | Extract tasks, owners, and deadlines from meeting notes |
| Python Code Generator | Produce clean and maintainable Python code              |
| Error Explainer       | Explain software errors and suggest fixes               |

---

## How It Works

### Beginner AI Explainer Workflow

```text
User Query
     │
     ▼
ChatPromptTemplate
     │
     ▼
OpenAI Model
     │
     ▼
Structured Explanation
     │
     ├── Simple Explanation
     ├── Technical Explanation
     ├── Interview Answer
     └── Real-World Use Cases
```

The application combines a predefined prompt template with a selected OpenAI model using LangChain's Expression Language (LCEL):

```python
chain = prompt | llm
response = chain.invoke({
    "user_query": user_query
})
```

---

## Getting Started

### Prerequisites

* Python 3.9+
* OpenAI API Key

### Installation

```bash
pip install langchain-openai langchain-core streamlit
```

### Run the Application

```bash
cd "Beginner AI Explainer"
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## Prompt Practice

The included prompt exercises demonstrate several core prompt engineering techniques:

### Role-Based Prompting

Assigning a specific expert role to the model.

Example:

```text
You are a senior Python software engineer.
```

### Structured Outputs

Forcing predictable output formats such as tables, sections, and bullet lists.

### Constraint-Based Prompting

Defining rules the model must follow.

Example:

```text
Do not invent information.
Summarize only what exists in the document.
```

### Task-Specific Prompt Design

Creating prompts optimized for:

* Summarization
* Information extraction
* Code generation
* Debugging assistance

---

## Learning Outcomes

After completing this module, you will understand:

* What Large Language Models are and how they work
* How to interact with LLMs using LangChain
* How to create reusable prompt templates
* How to structure prompts for reliable outputs
* How to build a simple LLM-powered application with Streamlit
* Basic LangChain Expression Language (LCEL) chaining
* Prompt engineering best practices for common AI tasks

---

## License

MIT
