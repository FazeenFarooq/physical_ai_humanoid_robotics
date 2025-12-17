"""
Health check API routes.
"""
from fastapi import APIRouter
from pydantic import BaseModel

# Initialize the router
health_router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    message: str


@health_router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return HealthResponse(status="healthy", message="RAG Chatbot API is running")