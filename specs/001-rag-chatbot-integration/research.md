# Research: RAG Chatbot for AI Book

## Decision: Embedding Model Selection
**Rationale:** Selected Cohere Embeddings API as required by the Integrated RAG Chatbot Constitution. Cohere embeddings provide high-quality semantic representations that work well for document retrieval scenarios.
**Alternatives considered:** OpenAI embeddings, Hugging Face sentence transformers, Qwen embeddings

## Decision: LLM Model Selection
**Rationale:** Selected Cohere Command/Command-R models as required by the constitution. These models provide excellent performance for question-answering tasks while supporting the system's safety requirements.
**Alternatives considered:** OpenAI GPT models, Anthropic Claude, OpenRouter-hosted models

## Decision: Vector Database
**Rationale:** Qdrant Cloud Free Tier selected as required by constitution. Offers efficient similarity search and good integration with Python ecosystem.
**Alternatives considered:** Pinecone, Weaviate, ChromaDB, FAISS

## Decision: Backend Framework
**Rationale:** FastAPI selected as required by constitution. Offers async support, automatic API documentation, and excellent performance for the use case.
**Alternatives considered:** Flask, Django, Starlette

## Decision: Database for Chat History
**Rationale:** Neon Serverless PostgreSQL selected as required by constitution. Provides reliable, ACID-compliant storage with serverless scalability.
**Alternatives considered:** Supabase, traditional PostgreSQL, MongoDB

## Decision: Frontend Integration
**Rationale:** Docusaurus integration with ChatKit SDK selected as required by constitution. Allows seamless embedding within the existing documentation site.
**Alternatives considered:** Custom React component, alternative chat SDKs

## Decision: Chunking Strategy
**Rationale:** Hierarchical chunking (chapter → section → paragraph) allows for flexible retrieval while maintaining context. Each chunk includes metadata for proper citation.
**Alternatives considered:** Fixed-size chunking, semantic chunking

## Decision: Retrieval Modes
**Rationale:** Implementing both selected-text-only and global retrieval modes to satisfy the context priority principle while providing flexibility.
**Alternatives considered:** Single retrieval mode, multi-document selection