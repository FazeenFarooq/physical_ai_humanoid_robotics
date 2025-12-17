from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from uuid import UUID
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Source(BaseModel):
    """
    Represents a source citation for the response.
    """
    chapter_id: str
    page_url: str
    content: str


class Message(BaseModel):
    """
    An individual exchange in the conversation, including the user's query 
    and the system's response.
    """
    id: UUID
    session_id: UUID
    role: MessageRole
    content: str
    created_at: datetime
    retrieved_chunks: Optional[Dict[str, Any]] = None  # IDs and metadata of chunks used for response
    retrieval_mode: Optional[str] = None  # "selected-text" or "global"
    sources: Optional[List[Source]] = None  # Citations to specific parts of the textbook
    
    class Config:
        from_attributes = True