from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel
from uuid import UUID


class VectorIndex(BaseModel):
    """
    A representation of the book content transformed into vectors for efficient similarity search.
    """
    chunk_id: str
    content: str
    embedding: list  # The vector representation of the content
    metadata: Dict[str, Any]  # Additional information including chapter_id, page_url, chunk_type
    created_at: datetime
    
    class Config:
        from_attributes = True

    @classmethod
    def create(cls, chunk_id: str, content: str, embedding: list, 
               chapter_id: str, page_url: str, chunk_type: str):
        """
        Create a new VectorIndex instance with default values.
        """
        valid_chunk_types = ["chapter", "section", "paragraph"]
        if chunk_type not in valid_chunk_types:
            raise ValueError(f"chunk_type must be one of {valid_chunk_types}")
        
        return cls(
            chunk_id=chunk_id,
            content=content,
            embedding=embedding,
            metadata={
                "chapter_id": chapter_id,
                "page_url": page_url,
                "chunk_type": chunk_type
            },
            created_at=datetime.utcnow()
        )