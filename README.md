# Customer Stories RAG Assistant

A Streamlit-based Retrieval-Augmented Generation (RAG) application that queries customer stories stored in Couchbase, powered by OpenAI embeddings and Nebius AI models.

## Overview

This project provides an interactive web interface for querying customer stories stored in a Couchbase database. It uses vector search to retrieve relevant documents and generates responses using a RAG pipeline with Nebius AI models. The application also optionally generates images based on user queries.

### Features
- **Vector Search**: Retrieves relevant documents from Couchbase using vector similarity search
- **RAG Pipeline**: Combines retrieved context with AI-generated responses
- **Interactive UI**: Streamlit-based chat interface with real-time responses
- **Image Generation**: Optional image generation based on queries
- **Configurable**: Adjustable temperature and result limits
- **Chat History**: Persistent conversation history during sessions

## Prerequisites

- Python 3.8+
- Couchbase Server with a configured bucket (`customer_stories`), scope (`stories`), and collection (`docs`)
- Full-text search index named `textembedding` with vector search capabilities
- OpenAI API key
- Nebius API key

## Install dependencies

```
pip install streamlit couchbase openai python-dotenv nest_asyncio
```

## Run the app

```
streamlit run app.py
```
