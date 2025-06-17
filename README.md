# Thesis Labs Playground

This repository is a personal sandbox for exploring concepts relevant to my data science thesis. It is organized as a series of small experiments and prototypes focused on large language models, Retrieval-Augmented Generation (RAG), LangChain, and document-based QA systems. The structure is modular, with each folder representing a different line of experimentation or conceptual test.

## Repository Overview

### pubhub_rag
A prototype Streamlit application that applies RAG techniques on scientific documents, specifically using data from PubHub. Includes notebooks for exploration, a Python app for interaction, and a basic document dataset. The focus here is on chat history referencing, chunking strategies, and document retrieval relevance.

### LangChain
A working environment to test LangChain’s capabilities including agents, custom chains, retrieval pipelines, and integration with various data formats. This folder includes multiple submodules and a requirements file to set up the environment.

### RAG
A minimal sandbox containing a Jupyter notebook used for isolated development and quick testing of RAG principles such as embedding models, prompt engineering, and vector similarity queries.

## Purpose

The goal of this repository is to freely test ideas, iterate over code, and gain hands-on understanding of modern LLM workflows. It serves as a practical foundation for designing and implementing thesis work related to applied AI systems and intelligent retrieval mechanisms.

## Technologies Explored

- Python
- Jupyter Notebooks
- LangChain framework
- OpenAI API and LLMs
- Streamlit
- Vector databases and embeddings (e.g., FAISS)
- Environment and dependency management

## Setup Notes

Each folder is mostly independent. To run any experiment:

1. Navigate to the folder of interest.
2. Install the required packages using the provided `requirements.txt` or `pkglist.txt`.
3. Create a `.env` file based on the `.env.example` if API keys are required.
4. Use Jupyter for notebooks or run the Streamlit app as documented.

## Author

Joan – exploring generative AI systems as part of a data science thesis.
