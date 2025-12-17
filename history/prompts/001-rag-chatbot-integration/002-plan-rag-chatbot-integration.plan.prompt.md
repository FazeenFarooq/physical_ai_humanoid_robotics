---
id: "002"
title: "plan-rag-chatbot-integration"
stage: "plan"
date_iso: "2025-12-16"
surface: "agent"
model: "Qwen"
feature: "001-rag-chatbot-integration"
branch: "001-rag-chatbot-integration"
user: "user"
command: "/sp.plan"
labels: ["planning", "rag", "chatbot", "ai-book"]
links: 
  spec: "../specs/001-rag-chatbot-integration/spec.md"
  ticket: null
  adr: null
  pr: null
files_yaml:
  - "specs/001-rag-chatbot-integration/plan.md"
  - "specs/001-rag-chatbot-integration/research.md"
  - "specs/001-rag-chatbot-integration/data-model.md"
  - "specs/001-rag-chatbot-integration/quickstart.md"
  - "specs/001-rag-chatbot-integration/contracts/openapi.yaml"
tests_yaml: []
prompt_text: "/sp.plan The project follows a three-phase architecture: Data Ingestion, Backend Core, and Frontend Integration.

Phase I: Data Ingestion Pipeline

Source Parsing: Load and extract clean text and metadata (chapter, section) from Docusaurus MDX files using Python tools (e.g., LangChain).

Chunking: Split the text into semantically cohesive chunks.

Qwen Embedding: Generate vectors for all chunks using the Qwen model.

Qdrant Indexing: Store vectors and associated metadata in a dedicated Qdrant collection.

Phase II: Backend Core Development (FastAPI Gateway)

FastAPI Initialization: Set up the project, dependencies, and environment variables (OpenRouter, Qdrant, Neon keys).

Neon DB Setup: Configure database connection and define ORM models for logging chat history and RAG parameters.

ChatKit Session Endpoint: Create the necessary endpoint for secure ChatKit frontend initialization and authentication.

RAG Core Endpoint: Implement the main API logic to handle the multi-source context (selected text + retrieved chunks).

Phase III: Real-Time Workflow & Frontend

Query Embedding: The FastAPI RAG endpoint uses the Qwen model to embed the incoming user query.

Retrieval: Use the Qwen vector to search Qdrant for the top K relevant chunks.

Prompt Construction: Construct the final, structured System Prompt, prioritizing the optional user-selected context over the retrieved chunks.

Generation (OpenRouter): Send the structured prompt via httpx to OpenRouter for the LLM response.

Docusaurus Integration: Embed the ChatKit UI component and implement a JavaScript listener to capture and pass the user's text selection to the RAG endpoint."
response_text: "Successfully completed the implementation planning workflow for the RAG Chatbot project. Created all required artifacts for Phase 0 (research) and Phase 1 (design & contracts) as follows:

Phase 0 - Research:
- Researched technology stack and best practices
- Documented decisions and alternatives for key components
- Created research.md with all findings

Phase 1 - Design & Contracts:
- Created data-model.md with entities and relationships
- Generated OpenAPI specification in contracts/openapi.yaml
- Created quickstart.md with setup and usage instructions
- Updated agent context with new technology stack

The implementation plan now covers all three phases of the architecture:
- Phase I: Data Ingestion Pipeline (MDX parsing, chunking, Qwen embeddings, Qdrant indexing)
- Phase II: Backend Core Development (FastAPI, Neon DB, ChatKit session, RAG endpoints)
- Phase III: Real-Time Workflow (Query embedding, retrieval, prompt construction with context prioritization, OpenRouter generation)

All required documentation and contracts are in place for proceeding to task breakdown and implementation."
outcome: "SUCCESS"
evaluation: "All planning phases completed successfully with proper research, design documentation, and API contracts generated. The Constitution Check passed with appropriate exceptions documented for a web-based system rather than a physical AI system."
---

# PHR: plan-rag-chatbot-integration

Planning for RAG Chatbot integration in AI textbook.

## Summary

Successfully completed the implementation planning workflow for the RAG Chatbot project. Created all required artifacts for Phase 0 (research) and Phase 1 (design & contracts) as follows:

Phase 0 - Research:
- Researched technology stack and best practices
- Documented decisions and alternatives for key components
- Created research.md with all findings

Phase 1 - Design & Contracts:
- Created data-model.md with entities and relationships
- Generated OpenAPI specification in contracts/openapi.yaml
- Created quickstart.md with setup and usage instructions
- Updated agent context with new technology stack

The implementation plan now covers all three phases of the architecture:
- Phase I: Data Ingestion Pipeline (MDX parsing, chunking, Qwen embeddings, Qdrant indexing)
- Phase II: Backend Core Development (FastAPI, Neon DB, ChatKit session, RAG endpoints)
- Phase III: Real-Time Workflow (Query embedding, retrieval, prompt construction with context prioritization, OpenRouter generation)

All required documentation and contracts are in place for proceeding to task breakdown and implementation.

## Files Created/Modified

- specs/001-rag-chatbot-integration/plan.md
- specs/001-rag-chatbot-integration/research.md
- specs/001-rag-chatbot-integration/data-model.md
- specs/001-rag-chatbot-integration/quickstart.md
- specs/001-rag-chatbot-integration/contracts/openapi.yaml

## Outcome

SUCCESS: All planning phases completed successfully with proper research, design documentation, and API contracts generated. The Constitution Check passed with appropriate exceptions documented for a web-based system rather than a physical AI system.