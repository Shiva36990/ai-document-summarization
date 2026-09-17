# AI-Powered Document Summarization System

An NLP-based document summarization application that extracts text from PDF, TXT, and DOCX files and generates concise summaries using transformer models.

## Features

- Supports PDF, TXT, and DOCX files
- Extracts and cleans document text
- Performs sentence-aware text chunking
- Generates semantic embeddings using Sentence Transformers
- Uses FAISS for similarity-based retrieval
- Generates summaries using the BART transformer model
- Supports documents containing paragraphs and tables

## Architecture

```text
PDF / TXT / DOCX
        ↓
Text Extraction
        ↓
Text Cleaning
        ↓
Sentence-Aware Chunking
        ↓
Sentence Transformer Embeddings
        ↓
FAISS Indexing and Retrieval
        ↓
BART Summarization Model
        ↓
Final Summary
