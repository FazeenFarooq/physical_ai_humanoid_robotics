<!-- SYNC IMPACT REPORT
Version change: 1.0.0 → 1.0.1 (updated for implementation flexibility)
Added sections: None
Removed sections: None
Modified principles: AI Stack (Embeddings and Generation sections updated to allow for implementation flexibility while maintaining core requirements)
Templates requiring updates: ✅ updated - .specify/templates/plan-template.md, .specify/templates/spec-template.md, .specify/templates/tasks-template.md
Follow-up TODOs: None
-->
# Integrated RAG Chatbot Constitution

## Core Principles

### Groundedness
All chatbot answers MUST be strictly derived from retrieved book content. The system must not generate responses based on external knowledge or internal training data alone. Every assertion in the response must trace back to a specific portion of the book's content.

### Non-hallucination
If relevant context is unavailable in the retrieved documents, the chatbot must respond with explicit uncertainty rather than fabricating information. Responses should acknowledge limitations and avoid making claims outside the provided context.

### Context Priority
User-selected text always overrides global retrieval when the "Selected-Text-Only" mode is activated. The system must constrain all processing and response generation to the explicitly selected text range provided by the user.

### Determinism
Given identical inputs (query + context), the system should yield semantically consistent outputs. While some variation is acceptable due to the probabilistic nature of LLMs, core meaning and factual content must remain stable.

### Transparency
The system must clearly indicate when answers are based on user-selected text versus global context retrieved from the broader book content. This helps establish trust and sets appropriate expectations.

## AI Stack (MANDATORY)

### Embeddings
All vector embeddings must utilize a high-quality API to maintain consistency and compatibility with the overall system architecture. (Current implementation uses Qwen embeddings as specified in requirements, but Cohere API is also acceptable per original constitution)

### Generation
Text generation must exclusively use a high-quality LLM API to ensure alignment with the project's AI technology choices and performance requirements. (Current implementation uses OpenRouter-hosted LLMs as specified in requirements, but Cohere Command/Command-R is also acceptable per original constitution)

### Backend
The backend must be implemented with FastAPI using async Python to handle concurrent requests efficiently and provide a clean RESTful interface.

### Vector Store
All document embeddings must be stored in Qdrant Cloud Free Tier for vector similarity search and retrieval operations.

### Database
Chat history, queries, retrieved document IDs, and responses must be stored in Neon Serverless PostgreSQL for persistence and analytics.

### Frontend
The user interface must be integrated directly into Docusaurus pages using custom integration with OpenAI Agents or ChatKit SDK as appropriate.

## Operational Rules

### Retrieval Precedence
Retrieval MUST occur before generation in every response cycle. The system is forbidden from generating answers without first retrieving relevant context from the book.

### Generation Constraints
Generation is strictly forbidden without retrieved context meeting minimum quality thresholds. The system must verify adequate context exists before attempting response generation.

### Selected-Text Mode
In "Selected-Text-Only" mode:
- Retrieval scope is restricted to user-selected text only
- No external or chapter-level context may be incorporated
- The system must clearly indicate to users when this mode is active

### Default Mode
In default mode:
- Retrieval occurs from chapter/page-level embeddings
- The system provides broader contextual understanding
- Users are informed about the scope of information being used

### Content Boundaries
The system must never introduce external knowledge, personal opinions, or unfounded assumptions. All responses must remain anchored to the book's content.

## Quality Standards

### Factual Precision
All responses must maintain high accuracy relative to the source material. Claims must be directly traceable to specific sections of the book content.

### Technical Language
Communication should use clear, concise technical language appropriate for readers with software engineering and AI backgrounds. Jargon should be used appropriately but defined when introducing new concepts.

### Target Audience
Content must be tailored for readers with software engineering and AI backgrounds, avoiding oversimplification while remaining accessible and educational.

### Conciseness
Responses should minimize verbosity and maximize informational signal. Eliminate redundant phrasing and focus on delivering value efficiently.

## Data & Logging

### Storage Requirements
Store all chat queries, retrieved document IDs, and generated responses in Neon Postgres database for analytics, improvement, and compliance purposes.

### Mode Tracking
Log whether "selected-text" or "global" mode was used for each query to understand usage patterns and improve the system.

### Privacy Protection
Ensure no personally identifiable information (PII) is leaked or stored inadvertently. Implement proper sanitization of user inputs where necessary.

### Audit Trail
Maintain comprehensive logging to support troubleshooting, performance analysis, and quality assurance activities.

## Failure Handling

### Low Retrieval Score
When retrieval scores fall below the established threshold, the system must respond with:
"The selected text / book content does not contain sufficient information to answer this question."

### Service Failures
The system must gracefully degrade when external services (Cohere API, Qdrant, etc.) are unavailable, providing clear messages to users about the temporary limitation.

### Error Recovery
Implement mechanisms to recover from transient failures without losing user context or requiring restart of conversations.

### Fallback Strategies
Provide alternative pathways or manual support options when automated systems cannot adequately address user queries.

## Compliance

### Lifecycle Adherence
Follow Spec-Kit Plus lifecycle strictly: /sp.specify → /sp.plan → /sp.tasks → /sp.implement. Do not auto-implement without explicit user confirmation.

### User Confirmation
Critical implementation steps require explicit user confirmation ("yes") before execution to prevent unwanted changes.

### Technology Restrictions
Adhere to the mandated tech stack without deviation unless explicitly approved through proper channels.

### Specification Alignment
All implementations must align with the original specification and core principles outlined in this constitution.

## Success Criteria

### Zero Hallucinations
Achieve zero instances of fabricated information in responses. Any uncertainty must be explicitly acknowledged rather than masked with false information.

### Accuracy Verification
Maintain high factual accuracy by consistently citing and basing responses on the provided book content.

### Context-Bounded Responses
Ensure all answers remain within the boundaries of the provided context, whether from selected text or global retrieval.

### Seamless Integration
Successfully embed the chatbot within Docusaurus pages without disrupting the reading experience.

### Performance Reliability
Maintain reliable performance when consuming Cohere APIs and responding within acceptable timeframes.

## Governance

This Constitution establishes the foundational principles governing all development, implementation, and operation of the Integrated RAG Chatbot project. All derivative materials, code, documentation, and implementations must align with these principles. 

Amendments to this Constitution require documentation of rationale, approval from project leadership, and a migration plan for existing components. All project artifacts must verify compliance with these principles. Any deviations must be justified and approved before implementation.

**Version**: 1.0.1 | **Ratified**: 2025-01-15 | **Last Amended**: 2025-12-17