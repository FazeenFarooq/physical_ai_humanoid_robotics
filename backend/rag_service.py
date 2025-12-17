"""
RAG (Retrieval-Augmented Generation) Service for the AI textbook chatbot.
This module implements the core RAG functionality using:
- Qwen embeddings for vectorization
- Qdrant Cloud for vector similarity search
- OpenRouter-hosted LLMs for generation
- Neon Serverless Postgres for storing chat history
"""
import asyncio
import logging
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from config import settings

# Import required libraries (these would be installed via requirements.txt)
try:
    import openai
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    import asyncpg
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    logging.error(f"Missing required dependencies: {e}")
    logging.error("Please install backend requirements: pip install -r backend/requirements.txt")


@dataclass
class DocumentChunk:
    """Represents a chunk of text from the textbook with its embedding."""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict


@dataclass
class RAGResponse:
    """Response from the RAG system."""
    answer: str
    sources: List[str]
    tokens_used: int
    processing_time: float
    confidence: float


class RAGService:
    """Main RAG service class."""

    def __init__(self):
        self.qdrant_client = None
        self.db_pool = None
        self.embedding_model = None
        self.llm_client = None
        self.collection_name = settings.qdrant_collection_name

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize all services: Qdrant, DB, Embedding model, LLM client."""
        try:
            # Initialize Qdrant client
            if settings.qdrant_url and settings.qdrant_api_key:
                self.qdrant_client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key
                )

                # Check if collection exists, create if not
                try:
                    self.qdrant_client.get_collection(self.collection_name)
                    self.logger.info(f"Qdrant collection '{self.collection_name}' exists")
                except:
                    self.logger.info(f"Creating Qdrant collection '{self.collection_name}'")
                    self.qdrant_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)  # Updated size for Qwen embeddings
                    )
            else:
                self.logger.warning("Qdrant configuration not provided, running without vector storage")

            # Initialize embedding model
            self.embedding_model = SentenceTransformer(settings.embedding_model)

            # Initialize OpenRouter client
            if settings.openrouter_api_key:
                self.llm_client = openai.AsyncOpenAI(
                    api_key=settings.openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1"
                )
            else:
                self.logger.warning("OpenRouter API key not provided, using mock responses")

            # Initialize database connection pool
            if settings.database_url:
                self.db_pool = await asyncpg.create_pool(settings.database_url)
            else:
                self.logger.warning("Database URL not provided, chat history will not be stored")

            self.logger.info("RAG service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG service: {e}")
            raise

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a text using the embedding model."""
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")

        embedding = self.embedding_model.encode([text])[0].tolist()
        return embedding

    async def store_document_chunks(self, chunks: List[DocumentChunk]):
        """Store document chunks in the vector database (Qdrant)."""
        if not self.qdrant_client:
            self.logger.warning("Qdrant client not initialized, skipping document storage")
            return

        points = []
        for chunk in chunks:
            points.append(
                models.PointStruct(
                    id=chunk.id,
                    vector=chunk.embedding,
                    payload={
                        "content": chunk.content,
                        "metadata": chunk.metadata
                    }
                )
            )

        # Upload points to Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        self.logger.info(f"Stored {len(chunks)} document chunks in Qdrant")

    async def search_similar_content(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar content in the vector database."""
        if not self.qdrant_client:
            self.logger.warning("Qdrant client not initialized, returning empty results")
            return []

        query_embedding = await self.embed_text(query)

        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )

        results = []
        for hit in search_result:
            results.append({
                "id": str(hit.id),
                "content": hit.payload["content"],
                "metadata": hit.payload["metadata"],
                "score": hit.score
            })

        return results

    async def generate_response(self, query: str, context: List[Dict], selected_context: str = None, model: str = None) -> str:
        """Generate response using the LLM with provided context."""
        if not self.llm_client:
            # Return mock response if LLM client not available
            self.logger.warning("LLM client not initialized, returning mock response")
            return f"This is a mock response for your query: '{query}'. In a real implementation, this would be generated by an LLM based on the provided context."

        if model is None:
            model = settings.openrouter_default_model

        # Format context for the LLM
        # Prioritize user-selected context over retrieved chunks (as per Phase III requirements)
        context_str = ""

        if selected_context:
            # User-selected context has higher priority
            context_str += f"IMPORTANT CONTEXT FROM USER SELECTION:\n{selected_context}\n\n"

        if context:
            # Retrieved chunks come next
            retrieved_content = "\n\n".join([f"[SOURCE: {item['id']}]\n{item['content'][:500]}" for item in context])
            context_str += f"ADDITIONAL CONTEXT FROM TEXTBOOK:\n{retrieved_content}"

        if not context_str:
            context_str = "No specific context provided. Answer based on general knowledge."

        # Construct the final prompt
        prompt = f"""
        You are an AI assistant for an AI textbook. Answer the user's question based on the provided context.
        IMPORTANT: Prioritize information from the "IMPORTANT CONTEXT FROM USER SELECTION" section over "ADDITIONAL CONTEXT FROM TEXTBOOK" when available.

        Context:
        {context_str}

        User's question: {query}

        Provide a clear, concise answer citing which sources you used. When citing sources, reference the source IDs in brackets like [SOURCE: ...].
        """

        try:
            response = await self.llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.7
            )

            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return "I encountered an error while generating the response. Please try again."

    async def store_chat_history(self, user_id: str, query: str, response: str, sources: List[str]):
        """Store the chat interaction in the database."""
        if not self.db_pool:
            self.logger.warning("Database pool not initialized, skipping chat history storage")
            return

        try:
            async with self.db_pool.acquire() as connection:
                await connection.execute(
                    """
                    INSERT INTO chat_history (user_id, query, response, sources, created_at)
                    VALUES ($1, $2, $3, $4, NOW())
                    """,
                    user_id, query, response, sources
                )
        except Exception as e:
            self.logger.error(f"Error storing chat history: {e}")

    async def get_chat_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Retrieve chat history for a user."""
        if not self.db_pool:
            self.logger.warning("Database pool not initialized, returning empty history")
            return []

        try:
            async with self.db_pool.acquire() as connection:
                rows = await connection.fetch(
                    """
                    SELECT query, response, sources, created_at
                    FROM chat_history
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    user_id, limit
                )

                history = []
                for row in rows:
                    history.append({
                        "query": row["query"],
                        "response": row["response"],
                        "sources": row["sources"],
                        "timestamp": row["created_at"]
                    })

                return history
        except Exception as e:
            self.logger.error(f"Error retrieving chat history: {e}")
            return []

    async def process_query(self, query: str, user_id: str = None, context_selection: str = None, model: str = None) -> RAGResponse:
        """Process a user query through the full RAG pipeline."""
        start_time = time.time()

        # Search for relevant content in the vector database if no context selection provided
        if not context_selection:
            retrieved_context = await self.search_similar_content(query)
        else:
            retrieved_context = await self.search_similar_content(query)

        # Generate response using the LLM (with priority to selected context)
        answer = await self.generate_response(
            query=query,
            context=retrieved_context,
            selected_context=context_selection,
            model=model
        )

        # Extract sources
        sources = [item["id"] for item in retrieved_context] if retrieved_context else []
        if context_selection:
            sources.append("user_selected")

        # Calculate metrics
        tokens_used = len(answer.split())
        processing_time = time.time() - start_time

        # Simple confidence based on number of sources and presence of selected context
        confidence = 0.5  # Base confidence
        if retrieved_context:
            confidence += min(0.3, len(retrieved_context) * 0.1)  # Up to 0.3 from retrieved content
        if context_selection:
            confidence += 0.2  # Bonus for selected context

        confidence = min(1.0, confidence)  # Cap at 1.0

        # Store in history if user_id provided
        if user_id:
            await self.store_chat_history(user_id, query, answer, sources)

        return RAGResponse(
            answer=answer,
            sources=sources,
            tokens_used=tokens_used,
            processing_time=processing_time,
            confidence=confidence
        )


# Global instance
rag_service = RAGService()