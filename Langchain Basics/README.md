# LangChain Basics

Learn the core concepts of LangChain through hands-on projects built with **LangChain**, **OpenAI**, and **Streamlit**.

This module covers prompt engineering, output parsing, structured outputs, sequential chains, parallel execution, query routing, and content generation workflows.

---

## Projects Included

### 1. AI Concept Explainer

Explains AI/ML concepts using layman-friendly, technical, and interview-oriented formats while filtering non-AI topics.

### 2. AI Topic Explainer

Generates structured explanations for AI/ML topics including definitions, use cases, components, examples, and summaries.

### 3. Blog Quiz Generator

Converts an article into a concise summary and automatically generates quiz questions from the summarized content.

### 4. Content Pipeline

Creates multiple content formats from a single topic, including a blog post, LinkedIn post, tweet, and summary.

### 5. Course Generator

Generates structured AI/ML course outlines using validated Pydantic schemas.

### 6. Parallel Document Analyzer

Performs summary generation, keyword extraction, sentiment analysis, and difficulty classification simultaneously.

### 7. Prompt Playground

Compares how different prompt instructions influence LLM responses for the same input.

### 8. Query Router

Classifies user queries into Coding, Technical, or General categories and routes them to specialized chains.

### 9. Research Assistant

Generates structured research reports in JSON format containing summaries, advantages, limitations, and future scope.

---

## Features

* Prompt Templates with LangChain
* LCEL (LangChain Expression Language)
* Sequential Chaining
* Parallel Execution
* Query Routing
* Structured Output Parsing
* Pydantic Validation
* JSON Output Generation
* Multi-Output Content Pipelines
* Streamlit-Based User Interfaces

---

## Tech Stack

| Layer          | Technology                                              |
| -------------- | ------------------------------------------------------- |
| Framework      | LangChain                                               |
| Models         | GPT-4o / GPT-4o-mini                                    |
| Frontend       | Streamlit                                               |
| Output Parsing | StrOutputParser, JsonOutputParser, PydanticOutputParser |
| Validation     | Pydantic                                                |
| Language       | Python 3.9+                                             |

---

## Project Structure

```text
LangChain Basics/
├── AI Concept Explainer/
│   └── app.py
├── AI Topic Explainer/
│   └── app.py
├── Blog_Quiz Generator/
│   └── app.py
├── Content Pipeline/
│   └── app.py
├── Course Generator/
│   └── app.py
├── Parallel Document Analyzer/
│   └── app.py
├── Prompt Playground/
│   └── app.py
├── Query Router/
│   └── app.py
└── Research Assistant/
    └── app.py
```

---

## Files

### `AI Concept Explainer/app.py`

Demonstrates prompt templates, output parsing, and AI-topic validation.

### `AI Topic Explainer/app.py`

Generates structured AI/ML topic explanations using custom prompt engineering.

### `Blog_Quiz Generator/app.py`

Implements a sequential workflow that summarizes an article and generates quiz questions.

### `Content Pipeline/app.py`

Uses multiple chains to create blog, LinkedIn, tweet, and summary content from a single topic.

### `Course Generator/app.py`

Uses `PydanticOutputParser` to generate validated course outlines.

### `Parallel Document Analyzer/app.py`

Uses `RunnableParallel` to execute multiple document analysis tasks concurrently.

### `Prompt Playground/app.py`

Demonstrates the impact of prompt design on LLM behavior.

### `Query Router/app.py`

Routes user queries to specialized chains based on LLM classification.

### `Research Assistant/app.py`

Produces structured research reports using JSON output parsing.

---

## How It Works

### AI Concept Explainer

```text
User Question → Prompt Template → LLM → Structured Explanation
```

### Blog Quiz Generator

```text
Article → Summarizer → Summary → Quiz Generator → Questions
```

### Content Pipeline

```text
Topic → Parallel Chains → Blog + LinkedIn + Tweet + Summary
```

### Course Generator

```text
Topic → LLM → Pydantic Parser → Structured Course Outline
```

### Parallel Document Analyzer

```text
Document → Parallel Tasks → Summary + Keywords + Sentiment + Difficulty
```

### Query Router

```text
User Query → Classifier → Router → Specialist Chain → Response
```

### Research Assistant

```text
Research Topic → LLM → JSON Parser → Structured Report
```

---

## Getting Started

### Prerequisites

* Python 3.9+
* OpenAI API Key

### Installation

```bash
pip install streamlit langchain langchain-openai langchain-core pydantic
```

---

## Running the Projects

### Run Any Project

```bash
cd "<project-folder>"
streamlit run app.py
```

Example:

```bash
cd "Query Router"
streamlit run app.py
```

Open the application at:

```text
http://localhost:8501
```

---

## Learning Outcomes

After completing this module, you will understand:

* Prompt Engineering Fundamentals
* ChatPromptTemplate Usage
* LangChain Expression Language (LCEL)
* Output Parsers
* Structured Outputs with Pydantic
* JSON Response Generation
* Sequential Chains
* Parallel Chains
* Query Routing
* Runnable Components
* Building AI Applications with Streamlit

---

## License

MIT