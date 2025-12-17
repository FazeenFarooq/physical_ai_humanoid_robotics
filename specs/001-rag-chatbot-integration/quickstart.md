# Quickstart: RAG Chatbot for AI Book

## Prerequisites

- Python 3.10+
- Access to Cohere API (for embeddings and generation)
- Qdrant Cloud account with API access
- Neon Serverless PostgreSQL account
- Node.js (for Docusaurus development)

## Setup

### 1. Environment Configuration

```bash
# Clone the repository
git clone <repo-url>
cd ai_textbook

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your actual API keys:
# COHERE_API_KEY=your_cohere_api_key
# QDRANT_URL=your_qdrant_cluster_url
# QDRANT_API_KEY=your_qdrant_api_key
# NEON_DATABASE_URL=your_neon_database_url
```

### 2. Database Setup

```bash
# Run database migrations (or set up tables manually)
# (Specific commands depend on your database setup script)
```

### 3. Initialize Vector Store

```bash
# First, generate embeddings for the book content:
python run_pipeline.py --action ingest

# This will:
# - Parse Docusaurus MDX files
# - Chunk the content hierarchically
# - Generate Cohere embeddings for each chunk
# - Store vectors in Qdrant with metadata
```

### 4. Run the Backend

```bash
# Start the FastAPI server
python main.py

# Or using the startup script:
./start_server.sh  # On Windows: start_server.bat
```

### 5. Frontend Integration

The Docusaurus site needs to be updated to include the chatbot widget. The widget should:
- Capture user questions
- Detect selected text on the page
- Send requests to the backend API
- Display responses with proper citations
- Indicate which mode (selected-text or global) was used

## API Usage

### Health Check
```
GET /health
```

### Chat Endpoint
```
POST /query
Content-Type: application/json

{
  "question": "Your question about the book content",
  "selected_text": "Optional selected text context",  # If empty, uses global retrieval
  "session_id": "Optional session ID" 
}
```

### Response Format
```json
{
  "response": "Answer to the question based on book content",
  "sources": [
    {
      "chapter_id": "Chapter identifier",
      "page_url": "URL of source page",
      "content": "Relevant text excerpt"
    }
  ],
  "retrieval_mode": "selected-text|global"
}
```

## Testing

Run the backend tests to verify functionality:
```bash
python -m pytest tests/
```

## Common Issues

1. **Cohere API access denied**: Verify your API key in the environment variables
2. **Qdrant connection errors**: Check your cluster URL and API key
3. **Response quality issues**: Verify the book content was properly ingested and indexed
4. **Slow responses**: Consider optimizing your chunking strategy or increasing timeout values