from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from uuid import UUID, uuid4


class ChatSession(BaseModel):
    """
    Represents a single user's conversation with the chatbot, containing metadata about the session.
    """
    id: UUID
    user_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime
    session_token: str
    
    class Config:
        from_attributes = True

    @classmethod
    def create(cls, user_id: Optional[UUID] = None):
        """
        Create a new ChatSession instance with default values.
        """
        now = datetime.utcnow()
        return cls(
            id=uuid4(),
            user_id=user_id,
            created_at=now,
            updated_at=now,
            session_token=str(uuid4())
        )