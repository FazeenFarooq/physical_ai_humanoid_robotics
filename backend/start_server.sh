#!/bin/bash
# Startup script for the RAG Chatbot FastAPI backend

echo "Starting RAG Chatbot backend service..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: venv not found, assuming global packages are installed"
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start the FastAPI application
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload