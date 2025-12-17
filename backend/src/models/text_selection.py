from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from uuid import UUID, uuid4


class TextSelection(BaseModel):
    """
    A portion of text from the AI book that the user has selected for context.
    """
    id: UUID
    session_id: UUID
    content: str
    page_url: str
    chapter_id: str
    created_at: datetime
    
    class Config:
        from_attributes = True

    @classmethod
    def create(cls, session_id: UUID, content: str, page_url: str, chapter_id: str):
        """
        Create a new TextSelection instance with default values.
        """
        if len(content) > 5000:
            raise ValueError("Content exceeds maximum character limit of 5000")
        
        if not page_url or not page_url.strip():
            raise ValueError("Page URL is required")
        
        return cls(
            id=uuid4(),
            session_id=session_id,
            content=content,
            page_url=page_url,
            chapter_id=chapter_id,
            created_at=datetime.utcnow()
        )