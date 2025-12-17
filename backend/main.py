"""
FastAPI backend for the RAG Chatbot in the AI textbook.
This implements the backend for the retrieval-augmented generation chatbot.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.v1 import chatbot_router
from .api.v1 import health_router
import uvicorn
import os


def create_app():
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AI Textbook RAG Chatbot API",
        description="API for the Retrieval-Augmented Generation Chatbot in the AI Textbook",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routers
    app.include_router(health_router, prefix="/api/v1", tags=["health"])
    app.include_router(chatbot_router, prefix="/api/v1", tags=["chatbot"])

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)