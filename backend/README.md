# RAG Chatbot Backend

This is the FastAPI backend for the Retrieval-Augmented Generation (RAG) chatbot in the AI textbook. The implementation follows a comprehensive three-phase architecture:

## Architecture Overview

```
Phase I: Data Ingestion Pipeline
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Docusaurus    │───▶│   Source        │───▶│   Qwen          │
│   MDX Files     │    │   Parsing &     │    │   Embedding     │
│                 │    │   Chunking      │    │   Vectors       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                              ┌─────────┴─────────┐
                                              │   Qdrant Cloud    │
                                              │ (Vector Database) │
                                              └───────────────────┘

Phase II: Backend Core Development
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │───▶│   FastAPI        │───▶│  OpenRouter     │
│   (Docusaurus)  │    │   Backend        │    │  (LLM)          │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Neon Postgres    │
                    │ (Chat History)    │
                    └───────────────────┘

Phase III: Real-Time Workflow
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Docusaurus    │───▶│   FastAPI RAG    │───▶│  OpenRouter     │
│   ChatKit UI    │    │   Endpoint       │    │  Generation     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   Qdrant Search   │
                    │ (Retrieval)       │
                    └───────────────────┘
```

## Three-Phase Implementation

### Phase I: Data Ingestion Pipeline
- **Source Parsing**: Load and extract clean text and metadata (chapter, section) from Docusaurus MDX files using Python tools
- **Chunking**: Split the text into semantically cohesive chunks
- **Qwen Embedding**: Generate vectors for all chunks using the Qwen model
- **Qdrant Indexing**: Store vectors and associated metadata in a dedicated Qdrant collection

### Phase II: Backend Core Development (FastAPI Gateway)
- **FastAPI Initialization**: Set up the project, dependencies, and environment variables
- **Neon DB Setup**: Configure database connection and define ORM models for logging chat history
- **ChatKit Session Endpoint**: Create the necessary endpoint for secure ChatKit frontend initialization and authentication
- **RAG Core Endpoint**: Implement the main API logic to handle the multi-source context

### Phase III: Real-Time Workflow & Frontend
- **Query Embedding**: The FastAPI RAG endpoint uses the Qwen model to embed the incoming user query
- **Retrieval**: Use the Qwen vector to search Qdrant for the top K relevant chunks
- **Prompt Construction**: Construct the final, structured System Prompt, prioritizing the optional user-selected context over the retrieved chunks
- **Generation (OpenRouter)**: Send the structured prompt to OpenRouter for the LLM response
- **Docusaurus Integration**: Prepare for embedding the ChatKit UI component and text selection capture

## Features

- **Question Answering**: Answers questions about the book content using RAG
- **Context Selection**: Answers questions based only on text selected by the user (with higher priority)
- **Source Citations**: Provides citations for responses to the textbook content
- **Chat History**: Stores conversations in Neon Serverless Postgres
- **Model Flexibility**: Supports multiple LLMs via OpenRouter (Qwen-2, Mistral, Llama)
- **Secure Session Management**: JWT-based session handling for ChatKit integration

## Installation

1. Install Python dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

2. Set up environment variables:
   ```bash
   cp backend/.env.example backend/.env
   # Edit backend/.env with your API keys and configuration
   ```

3. Run the complete pipeline:
   ```bash
   cd backend
   python run_pipeline.py
   ```

4. Or run the application separately:
   ```bash
   cd backend
   python -m uvicorn main:app --reload
   ```

## Environment Variables

Create a `.env` file in the backend directory with the following variables:

```bash
# Qdrant Configuration
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

# OpenRouter Configuration
OPENROUTER_API_KEY=your_openrouter_api_key

# Database Configuration
DATABASE_URL=postgresql://user:password@host:port/database
```

## API Endpoints

### Core RAG Endpoints
- `GET /api/v1/health` - Health check
- `POST /api/v1/chat` - Main RAG chat endpoint
- `GET /api/v1/history/{user_id}` - Get chat history for a user
- `GET /api/v1/models` - Get available models

### ChatKit Session Endpoints
- `POST /api/v1/session` - Create a new session
- `POST /api/v1/session/validate` - Validate a session token
- `POST /api/v1/session/end` - End a session

### Chat Endpoint

Request:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is a neural network?"
    }
  ],
  "context_selection": "Optional text selected by user (higher priority)",
  "user_id": "Optional user ID for history tracking",
  "model": "qwen-2"
}
```

Response:
```json
{
  "response": "A neural network is...",
  "sources": ["Chapter 3", "user_selected"],
  "tokens_used": 150,
  "processing_time": 1.23,
  "confidence": 0.85
}
```

## Running the Complete Pipeline

### Phase I: Ingestion
To process your textbook content into the RAG system:

```python
from backend.ingestion import DataIngestionPipeline
import asyncio

async def main():
    # Initialize the RAG service first
    from backend.rag_service import rag_service
    await rag_service.initialize()

    # Create and run the ingestion pipeline
    pipeline = DataIngestionPipeline()
    await pipeline.run_pipeline("path/to/your/textbook/docs/")

asyncio.run(main())
```

### Phase II: Backend Core
The backend includes all necessary components for the complete workflow:

```bash
# Run the complete pipeline (includes all 3 phases)
python run_pipeline.py

# Or run just the backend server
python -m uvicorn main:app --reload
```

## Development

1. Make sure you have Python 3.8+ installed
2. Install dependencies: `pip install -r backend/requirements.txt`
3. Set up your environment variables
4. Run the complete pipeline: `python run_pipeline.py`
5. Or run the development server: `python -m uvicorn main:app --reload`

## Running the Server

Windows:
```bash
backend\start_server.bat
```

Linux/Mac:
```bash
chmod +x backend/start_server.sh
backend/start_server.sh
```

The server will be available at `http://localhost:8000`.

## Integration with Frontend

The backend is designed to be integrated with the Docusaurus frontend via ChatKit SDK. The API follows REST principles and returns JSON responses that the frontend can easily consume. The complete architecture supports real-time text selection capture and prioritization in the RAG pipeline.