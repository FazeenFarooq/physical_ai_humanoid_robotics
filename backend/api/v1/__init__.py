"""
API routes for the RAG Chatbot.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import logging

# Initialize the router
chatbot_router = APIRouter()

# Import the health router
from .health_router import health_router

# Import the RAG service
from ...rag_service import rag_service

# Import session manager for ChatKit integration
from ...session_manager import session_manager, SessionToken

# Pydantic models for request/response
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    context_selection: Optional[str] = None  # User-selected text context
    user_id: Optional[str] = None  # For tracking chat history
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    model: Optional[str] = None  # LLM model to use


class ChatResponse(BaseModel):
    response: str
    sources: List[str]  # Citations for the response
    tokens_used: int
    processing_time: float
    confidence: float


class ChatHistoryRequest(BaseModel):
    user_id: str


@chatbot_router.on_event("startup")
async def startup_event():
    """Initialize the RAG service when the app starts."""
    await rag_service.initialize()


@chatbot_router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """
    Main chat endpoint for the RAG chatbot.
    """
    try:
        # Get the last user message
        user_message = chat_request.messages[-1].content if chat_request.messages else ""

        # Process the query through the RAG pipeline
        rag_response = await rag_service.process_query(
            query=user_message,
            user_id=chat_request.user_id,
            context_selection=chat_request.context_selection,
            model=chat_request.model
        )

        return ChatResponse(
            response=rag_response.answer,
            sources=rag_response.sources,
            tokens_used=rag_response.tokens_used,
            processing_time=rag_response.processing_time,
            confidence=rag_response.confidence
        )
    except Exception as e:
        logging.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@chatbot_router.get("/history/{user_id}")
async def get_history(user_id: str):
    """
    Retrieve chat history for a specific user.
    """
    try:
        history = await rag_service.get_chat_history(user_id)
        return {"user_id": user_id, "history": history}
    except Exception as e:
        logging.error(f"Error retrieving chat history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@chatbot_router.get("/models")
async def get_available_models():
    """
    Get list of available models for the RAG system.
    """
    # In a real implementation, this would return models from OpenRouter
    return {
        "models": [
            {"id": "qwen-2", "name": "Qwen-2", "description": "Qwen model from OpenRouter"},
            {"id": "mistral", "name": "Mistral", "description": "Mistral model from OpenRouter"},
            {"id": "llama", "name": "Llama", "description": "Llama model from OpenRouter"}
        ]
    }


# ChatKit Session Endpoints
@chatbot_router.post("/session", response_model=SessionToken)
async def create_session(user_id: Optional[str] = None):
    """
    Create a new ChatKit session for the frontend.
    """
    try:
        session_token = await session_manager.create_session(user_id)
        return session_token
    except Exception as e:
        logging.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")


@chatbot_router.post("/session/validate")
async def validate_session(token: str):
    """
    Validate a ChatKit session token.
    """
    try:
        session_info = await session_manager.validate_session(token)
        return {"valid": True, "session_info": session_info}
    except Exception as e:
        logging.error(f"Error validating session: {e}")
        raise HTTPException(status_code=401, detail="Invalid session")


@chatbot_router.post("/session/end")
async def end_session(session_id: str):
    """
    End a ChatKit session.
    """
    try:
        await session_manager.end_session(session_id)
        return {"message": "Session ended successfully"}
    except Exception as e:
        logging.error(f"Error ending session: {e}")
        raise HTTPException(status_code=500, detail="Failed to end session")