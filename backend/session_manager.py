"""
ChatKit Session Endpoint for the RAG Chatbot
Implements secure session initialization and authentication for the frontend
"""
import uuid
import jwt
import time
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from config import settings


class SessionToken(BaseModel):
    session_id: str
    token: str
    expires_at: str


class SessionManager:
    """Manages chat sessions and authentication."""
    
    def __init__(self):
        # In a real implementation, you'd store active sessions in a database or cache
        # For now, we'll use an in-memory structure
        self.active_sessions = {}
        self.secret_key = settings.openrouter_api_key or "fallback_secret_key"
    
    async def create_session(self, user_id: Optional[str] = None) -> SessionToken:
        """Create a new authenticated session for a user."""
        session_id = str(uuid.uuid4())
        
        # Create JWT token
        payload = {
            "session_id": session_id,
            "user_id": user_id or "anonymous",
            "exp": datetime.utcnow() + timedelta(hours=24),  # Session expires in 24 hours
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        
        # Store session info
        self.active_sessions[session_id] = {
            "user_id": user_id or "anonymous",
            "created_at": datetime.utcnow(),
            "expires_at": payload["exp"]
        }
        
        return SessionToken(
            session_id=session_id,
            token=token,
            expires_at=payload["exp"].isoformat()
        )
    
    async def validate_session(self, token: str) -> dict:
        """Validate a session token and return session info."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            
            session_id = payload.get("session_id")
            if not session_id or session_id not in self.active_sessions:
                raise HTTPException(status_code=401, detail="Invalid session")
            
            # Check if session has expired
            if datetime.utcnow() > self.active_sessions[session_id]["expires_at"]:
                del self.active_sessions[session_id]
                raise HTTPException(status_code=401, detail="Session expired")
            
            return {
                "session_id": session_id,
                "user_id": payload.get("user_id")
            }
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    async def end_session(self, session_id: str):
        """End and remove a session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]


# Global session manager instance
session_manager = SessionManager()