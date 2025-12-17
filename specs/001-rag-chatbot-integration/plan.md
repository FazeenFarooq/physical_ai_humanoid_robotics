# Implementation Plan: RAG Chatbot for AI Book

**Branch**: `001-rag-chatbot-integration` | **Date**: 2025-12-16 | **Spec**: [link]
**Input**: Feature specification from `/specs/001-rag-chatbot-integration/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/scripts/powershell/` for the execution workflow.

## Summary

Implementation of a Retrieval-Augmented Generation (RAG) chatbot for the AI textbook following the high-level architecture plan, ensuring strict adherence to the Integrated RAG Chatbot Constitution:

**Phase I: Data Ingestion Pipeline**
- Source parsing: Load and extract clean text and metadata (chapter, section) from Docusaurus MDX files
- Chunking: Split the text into semantically cohesive chunks using chapter → section → paragraph hierarchy
- Embedding: Generate vectors for all chunks using Cohere Embeddings API (aligning with constitution)
- Qdrant Indexing: Store vectors and associated metadata (chapter_id, page_url, chunk_type) in a dedicated Qdrant collection
- Metadata storage: Store chapter_id, page_url, and chunk_type (chapter/section/paragraph) for proper citation

**Phase II: Backend Core Development (FastAPI Gateway)**
- FastAPI Initialization: Web framework setup with proper routing and error handling
- Cohere Integration: Implement Cohere Embeddings API for all vector operations
- Cohere LLM Integration: Implement Cohere Command/Command-R models for generation
- Neon DB Setup: Configure database connection for chat history storage
- API Endpoints:
  - POST /query (main RAG endpoint with selected-text detection)
  - GET /health (system health)
- Input validation and selected-text detection
- Orchestration of retrieval + generation with strict RAG constraints

**Phase III: Real-Time Workflow & Frontend**
- Query Processing: Detect selected-text presence and determine retrieval mode
- Embedding Generation: Vectorize user questions and selected text using Cohere
- Retrieval: Search Qdrant for relevant chunks based on retrieval mode:
  - Selected-text-only mode: Restrict to user-selected text only
  - Global mode: Use broader book context
- Prompt Construction: Build system prompts prioritizing user-selected context over retrieved content
- Generation: Use Cohere LLMs with strict system instructions preventing hallucination
- Response Validation: Verify responses adhere to RAG rules before returning
- Frontend Integration: Implement Docusaurus widget capturing user questions and selected text
- Citation Display: Clearly indicate source material for all generated content
- Mode Indication: Show user which retrieval mode is active (selected-text vs global)

The system is built with Python, FastAPI, Qdrant, Cohere APIs, and Neon Postgres as specified in the constitution, ensuring zero hallucination paths and explicit enforcement of selected-text logic.

## Technical Context

**Language/Version**: Python 3.10+ (as required by project setup)
**Primary Dependencies**:
  - FastAPI (web framework) - aligns with constitution requirement
  - Qdrant (vector database) - aligns with constitution requirement
  - Cohere Python SDK (LLM and embedding integration) - aligns with constitution
  - asyncpg (PostgreSQL async driver)
  - PyJWT (authentication)
  - markdown, beautifulsoup4 (MDX/HTML parsing)
**Storage**:
  - Vector storage: Qdrant Cloud - aligns with constitution requirement
  - Chat history: Neon Serverless Postgres - aligns with constitution requirement
  - Document metadata: stored in Qdrant payload with chapter_id, page_url, chunk_type
**Testing**: pytest (unit and integration testing)
**Target Platform**: Linux server (backend), Web browser (frontend via Docusaurus)
**Project Type**: Web application (backend API + Docusaurus frontend integration)
**Performance Goals**:
  - 90% of user questions answered within 5 seconds (per spec SC-001)
  - Response accuracy of 85%+ (per spec SC-004)
  - Support 100+ concurrent users (per spec SC-006)
**Constraints**:
  - Must use Cohere Embeddings API for vectorization (per constitution)
  - Must use Cohere Command/Command-R models for generation (per constitution)
  - Must use Qdrant Cloud for similarity search (per constitution)
  - Must use Neon Serverless Postgres for chat history (per constitution)
  - Must implement Docusaurus frontend integration (per constitution)
  - Answers must be sourced only from textbook content (Groundedness principle)
  - Responses must prioritize user-selected text context (Context Priority principle)
  - Generation forbidden without retrieved context meeting quality thresholds (Operational Rule)
  - Zero hallucinated answers required (Success Criteria)
**Scale/Scope**:
  - Support entire AI textbook content
  - Handle multiple concurrent users asking various questions
  - Store extensive chat history for returning users

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Gates determined based on Integrated RAG Chatbot Constitution:

- **Groundedness**: ✅ APPLICABLE - All responses must derive strictly from retrieved book content (per spec FR-002)
- **Non-hallucination**: ✅ APPLICABLE - System must respond with uncertainty if relevant context is unavailable (per spec FR-004)
- **Context priority**: ✅ APPLICABLE - User-selected text must override global retrieval when in selected-text mode (per spec FR-003)
- **Determinism**: ✅ APPLICABLE - Same inputs should yield semantically consistent outputs as specified in the architecture
- **Transparency**: ✅ APPLICABLE - System must indicate when answers are based on selected text vs global context
- **Uses Cohere Embeddings API for all vector embeddings**: ✅ APPLICABLE - Using Cohere embeddings as required by constitution
- **Uses Cohere Command/Command-R models for text generation**: ✅ APPLICABLE - Using Cohere Command/Command-R models as required by constitution
- **Backend implemented with FastAPI (async, Python)**: ✅ APPLICABLE - FastAPI backend being implemented as specified
- **Vector store implemented with Qdrant Cloud Free Tier**: ✅ APPLICABLE - Qdrant Cloud being used for vector storage as specified
- **Database implemented with Neon Serverless PostgreSQL**: ✅ APPLICABLE - Neon Postgres used for chat history as specified
- **Frontend integrated into Docusaurus with custom backend integration**: ✅ APPLICABLE - Integration with Docusaurus being implemented as specified
- **Retrieval occurs before generation in every response**: ✅ APPLICABLE - Core RAG workflow implemented as specified
- **Generation forbidden without retrieved context meeting quality thresholds**: ✅ APPLICABLE - Implemented with fallback responses when retrieval score is low (per spec FR-006)
- **Selected-text mode restricts retrieval to user-selected text only**: ✅ APPLICABLE - Implementation designed to support selected-text-only mode (per spec FR-003)
- **Zero hallucinated answers achieved**: ✅ APPLICABLE - Core project goal to prevent hallucinations (per spec FR-002)
- **Accurate, context-bound responses maintained**: ✅ APPLICABLE - Core project goal to maintain accuracy (per spec FR-002)
- **Seamless embedding inside Docusaurus pages**: ✅ APPLICABLE - Integration with Docusaurus planned as specified
- **Reliable performance on Cohere APIs**: ✅ APPLICABLE - Using Cohere APIs as required by constitution
- **Fallback to "insufficient information" response when retrieval score < threshold**: ✅ APPLICABLE - Implemented as specified per failure handling requirements

### Constitution Check Gate Result: PASSED

The RAG chatbot project fully aligns with all Integrated RAG Chatbot Constitution principles. All technology choices now comply with the constitution requirements: using Cohere Embeddings API, Cohere Command/Command-R models, FastAPI backend, Qdrant Cloud for vector storage, Neon Serverless PostgreSQL for chat history, and Docusaurus frontend integration.

### Re-check After Phase 1 Design:
To be updated after Phase 1 design completion - expected to remain PASSED as design aligns with constitution requirements.

## Project Structure

### Documentation (this feature)

```text
specs/001-rag-chatbot-integration/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
│   └── openapi.yaml     # API contract definition
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

# Web application (backend API + Docusaurus frontend integration)

backend/
├── main.py              # FastAPI application entry point
├── config.py            # Configuration settings using Pydantic
├── rag_service.py       # Core RAG implementation
├── session_manager.py   # JWT-based session management
├── run_pipeline.py      # Complete pipeline runner
├── start_server.bat     # Windows startup script
├── start_server.sh      # Linux/Mac startup script
├── requirements.txt     # Python dependencies
├── database_schema.sql  # Postgres schema definition
├── api/
│   └── v1/
│       ├── __init__.py  # API routes (chat, history, models, session)
│       └── health_router.py  # Health check endpoint
├── ingestion/
│   ├── __init__.py      # MDX parsing, chunking & embedding pipeline
├── tests/               # Unit and integration tests
└── README.md            # Comprehensive documentation

**Structure Decision**: The project follows a web application architecture with a dedicated backend for the RAG functionality. The existing Docusaurus frontend remains unchanged, with the backend providing API endpoints for the RAG chatbot. The backend includes all necessary components for the three-phase architecture: data ingestion pipeline, core RAG service, and API endpoints. Documentation includes research findings, data models, quickstart guide, and API contracts as required by the planning workflow.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No complexity violations requiring justification - the Constitution Check passed completely with all requirements now properly addressed.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [current need] | [why direct DB access insufficient] |

## Phase 1 Completion Summary

Phase 1 of the planning has been completed with the following artifacts generated:

- **research.md**: Resolved all technical unknowns and provided rationale for technology choices
- **data-model.md**: Defined all necessary data entities and their relationships
- **quickstart.md**: Created comprehensive quickstart guide for developers
- **contracts/openapi.yaml**: Specified API contracts for backend endpoints
- **Updated agent context**: Configured with new technologies (not applicable for manual workflow)

All Phase 1 deliverables completed and aligned with the Integrated RAG Chatbot Constitution requirements.

### Re-check After Phase 1 Design - Updated:

PASSED - The design fully aligns with constitution requirements and Phase 1 artifacts confirm technical feasibility of the implementation approach.
