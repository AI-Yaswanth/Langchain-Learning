# Memory & Conversation

Learn how to build stateful AI applications using LangChain memory components, conversation history management, and token-aware context handling.

This module focuses on maintaining conversational context across interactions, managing session-based memory, and implementing scalable memory strategies for long-running AI conversations.

---

## Skills & Concepts Covered

* Conversation Memory
* Chat History Management
* Session-Based Conversations
* RunnableWithMessageHistory
* MessagesPlaceholder
* InMemoryChatMessageHistory
* Context Retention
* Multi-Turn Conversations
* Token Counting
* Memory Compression
* Conversation Summarization
* Context Window Optimization
* Stateful AI Applications
* Streamlit Integration

---

## Projects Included

| Project                                       | Skills Demonstrated                                           |
| --------------------------------------------- | ------------------------------------------------------------- |
| **StudyBuddy – In-Memory Learning Companion** | Session Memory, Context-Aware Tutoring, Multi-Turn Learning   |
| **Session-Aware Conversational Chatbot**      | Conversation History Management, Session Isolation            |
| **Adaptive Token-Aware Memory Manager**       | Memory Compression, Token Tracking, Long-Context Optimization |

---

## Advanced LangChain Patterns Implemented

### Session-Based Memory

Maintains independent conversation history for multiple user sessions.

```text
User Message
      ↓
 Session History
      ↓
 Prompt + History
      ↓
 Context-Aware Response
```

### Conversational Context Retention

Preserves previous interactions and uses them to generate more relevant follow-up responses.

```text
Question 1
     ↓
 Memory Store
     ↓
 Question 2
     ↓
 Context-Aware Answer
```

### Multi-Session Isolation

Separates conversation history across different user sessions.

```text
Session 1 → Independent Memory
Session 2 → Independent Memory
Session 3 → Independent Memory
```

### Token-Aware Memory Compression

Compresses older conversation history into summaries when token usage exceeds predefined thresholds.

```text
Conversation History
         ↓
    Token Counter
         ↓
 Threshold Exceeded?
         ↓
    Summarization
         ↓
 Compressed Memory
```

---

## Project Structure

```text
Memory & Conversation/
├── StudyBuddy - In-Memory Learning Companion/
│   └── app.py
├── Session-Aware Conversational Chatbot/
│   └── app.py
└── Adaptive Token-Aware Memory Manager/
    └── app.py
```

---

## Files

### `StudyBuddy - In-Memory Learning Companion/app.py`

A tutoring assistant that remembers previously discussed concepts within a study session and adapts explanations based on the learner's progress.

**Key Concepts:**

* RunnableWithMessageHistory
* Session-based memory
* Context-aware tutoring
* Multi-turn educational conversations

---

### `Session-Aware Conversational Chatbot/app.py`

A general-purpose chatbot that maintains conversation history separately for each session.

**Key Concepts:**

* Chat history persistence
* Session isolation
* Context retention
* Conversational continuity

---

### `Adaptive Token-Aware Memory Manager/app.py`

A memory management system that tracks token usage and automatically compresses older conversation history through summarization when memory limits are reached.

**Key Concepts:**

* Token counting with tiktoken
* Memory compression
* Conversation summarization
* Context window optimization

---

## Tech Stack

| Layer             | Technology                 |
| ----------------- | -------------------------- |
| AI Framework      | LangChain                  |
| Models            | GPT-4o, GPT-4o-mini        |
| Frontend          | Streamlit                  |
| Memory Management | InMemoryChatMessageHistory |
| Memory Wrapper    | RunnableWithMessageHistory |
| Token Analysis    | tiktoken                   |
| Workflow Design   | LCEL                       |
| Language          | Python                     |

---

## Getting Started

### Prerequisites

* Python 3.9+
* OpenAI API Key

### Installation

```bash
pip install streamlit langchain langchain-openai langchain-core tiktoken
```

### Run a Project

```bash
cd "<project-folder>"
streamlit run app.py
```

Example:

```bash
cd "StudyBuddy - In-Memory Learning Companion"
streamlit run app.py
```

---

## What I Learned

Through these projects, I gained hands-on experience with:

### Conversational AI

* Building stateful chat applications
* Managing multi-turn conversations
* Maintaining context across interactions
* Designing memory-aware AI systems

### LangChain Memory

* RunnableWithMessageHistory
* MessagesPlaceholder
* InMemoryChatMessageHistory
* Session-based memory management

### Long-Context Strategies

* Token usage monitoring
* Memory compression techniques
* Conversation summarization
* Context-window optimization

### AI Application Development

* Multi-session architectures
* User-specific memory handling
* Interactive Streamlit interfaces
* Production-oriented conversation workflows

---

## License

MIT

