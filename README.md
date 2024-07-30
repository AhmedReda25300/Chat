# Chat with Websites

This Streamlit application allows users to interact with websites or PDF documents through a chat interface. It uses LangChain and Google Generative AI embeddings to process text and provide responses based on the content of the documents.

## Features

- Chat with websites using a URL.
- Upload multiple PDF files and interact with their content.
- Utilize Google Generative AI for embeddings and language generation.
- Retry mechanism for handling temporary issues.

## Prerequisites

Ensure you have the following software installed:

- Python 3.7 or later
- Miniconda or Python environment

## Installation

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd <repository-directory>

## Create a Virtual Environment
- conda create --name chat-app python=3.7
- conda activate chat-app


1. **Install Required Packages**

   ```bash
   pip install -r requirements.txt

## Set Up Environment Variables
- GOOGLE_EMBEDDING_MODEL=your-google-embedding-model-id

# Usage
streamlit run app.py
