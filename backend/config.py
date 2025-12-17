"""
Configuration settings for the RAG Chatbot backend.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API Settings
    app_name: str = "AI Textbook RAG Chatbot API"
    api_version: str = "v1"
    debug: bool = False
    
    # Database settings (Neon Serverless Postgres)
    database_url: Optional[str] = None
    
    # Qdrant settings (Vector database)
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "textbook_content"
    
    # Cohere settings (for LLMs)
    cohere_api_key: Optional[str] = None
    cohere_default_model: str = "command-r"

    # Embedding settings (Cohere embeddings)
    embedding_model: str = "embed-english-v3.0"
    
    # Frontend URL for CORS
    frontend_url: str = "http://localhost:3000"
    
    class Config:
        env_file = ".env"


settings = Settings()