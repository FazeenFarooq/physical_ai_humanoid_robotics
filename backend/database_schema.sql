-- Database schema for the RAG Chatbot
-- Using PostgreSQL syntax for Neon Serverless

-- Table for storing chat history
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    sources TEXT[], -- Array of source identifiers
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for efficient retrieval by user_id and timestamp
CREATE INDEX IF NOT EXISTS idx_chat_history_user_timestamp 
ON chat_history (user_id, created_at DESC);

-- Table for storing textbook content chunks (for RAG)
CREATE TABLE IF NOT EXISTS textbook_chunks (
    id VARCHAR(255) PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(4096), -- Assuming 4096-dim embeddings from Qwen
    chapter VARCHAR(255),
    section VARCHAR(255),
    page_number INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for embedding similarity search (this would be handled by Qdrant in our case)
-- We still create it for completeness if we need to fall back to DB-based similarity
CREATE INDEX IF NOT EXISTS idx_textbook_chunks_embedding 
ON textbook_chunks 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Table for tracking document processing status
CREATE TABLE IF NOT EXISTS document_processing (
    id SERIAL PRIMARY KEY,
    document_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'processing', -- processing, completed, failed
    chunks_processed INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Example query for finding similar content:
-- This would typically be done in Qdrant, but here's the Postgres equivalent
-- SELECT content, chapter, section FROM textbook_chunks
-- ORDER BY embedding <=> '[0.1,0.2,0.3,...]' -- cosine distance to query embedding
-- LIMIT 5;