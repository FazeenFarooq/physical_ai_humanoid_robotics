---
id: ca0963f7-cc1d-42b9-9978-cf382cc6f9e9
title: RAG Chatbot Implementation Plan
stage: plan
date: 2025-12-17
surface: agent
model: gpt-4o
feature: rag-chatbot
branch: 001-rag-chatbot-integration
user: qwen-user
command: /sp.plan
labels: ["plan", "rag", "chatbot", "docusaurus", "cohere"]
links:
  spec: /specs/001-rag-chatbot-integration/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-rag-chatbot-integration/plan.md
  - specs/001-rag-chatbot-integration/research.md
  - specs/001-rag-chatbot-integration/data-model.md
  - specs/001-rag-chatbot-integration/quickstart.md
  - specs/001-rag-chatbot-integration/contracts/openapi.yaml
tests: []
---

## Prompt

---
description: Execute the implementation planning workflow using the plan template to generate design artifacts.
---

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

1. **Setup**: Run `.specify/scripts/powershell/setup-plan.ps1 -Json` from repo root and parse JSON for FEATURE_SPEC, IMPL_PLAN, SPECS_DIR, BRANCH. For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

2. **Load context**: Read FEATURE_SPEC and `.specify/memory/constitution.md`. Load IMPL_PLAN template (already copied).

3. **Execute plan workflow**: Follow the structure in IMPL_PLAN template to:
   - Fill Technical Context (mark unknowns as "NEEDS CLARIFICATION")
   - Fill Constitution Check section from constitution
   - Evaluate gates (ERROR if violations unjustified)
   - Phase 0: Generate research.md (resolve all NEEDS CLARIFICATION)
   - Phase 1: Generate data-model.md, contracts/, quickstart.md
   - Phase 1: Update agent context by running the agent script
   - Re-evaluate Constitution Check post-design

4. **Stop and report**: Command ends after Phase 2 planning. Report branch, IMPL_PLAN path, and generated artifacts.

## Phases

### Phase 0: Outline & Research

1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:

   ```text
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

### Phase 1: Design & Contracts

**Prerequisites:** `research.md` complete

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Agent context update**:
   - Run `.specify/scripts/powershell/update-agent-context.ps1 -AgentType qwen`
   - These scripts detect which AI agent is in use
   - Update the appropriate agent-specific context file
   - Add only new technology from current plan
   - Preserve manual additions between markers

**Output**: data-model.md, /contracts/*, quickstart.md, agent-specific file

## Key rules

- Use absolute paths
- ERROR on gate failures or unresolved clarifications

---

/sp.plan /sp.plan

Objective:
Create a clear, step-by-step implementation plan for an Integrated
Retrieval-Augmented Generation (RAG) Chatbot embedded inside a
Docusaurus-published AI-native book, strictly following the constraints
and principles defined in /sp.constitution and /sp.specify.

High-Level Architecture Plan:

1. Frontend Layer (Docusaurus + ChatKit SDK)
   - Embed a chat widget on book pages.
   - Capture:
     a) User question
     b) Optional user-selected text from the page
   - Send structured requests to backend API.
   - Display responses with clear indication of:
     - Selected-text mode
     - Global book context mode

2. Backend Layer (FastAPI)
   - Expose REST API endpoints:
     - POST /query (main RAG endpoint)
     - GET /health (system health)
   - Responsibilities:
     - Validate input
     - Detect selected-text presence
     - Orchestrate retrieval + generation
     - Enforce RAG constraints (no hallucination)

3. Retrieval Layer (Qdrant Cloud)
   - Store vector embeddings of book content.
   - Chunking strategy:
     - Chapter → Section → Paragraph
   - Metadata to store:
     - chapter_id
     - page_url
     - chunk_type (chapter / section / paragraph)
   - Retrieval modes:
     a) Selected-text-only retrieval
     b) Global book retrieval (fallback)

4. Embedding Layer (Cohere)
   - Use Cohere Embeddings API.
   - Embed:
     - Book content during ingestion
     - User-selected text (runtime)
     - User questions (runtime)
   - Ensure consistent embedding model across system.

5. Generation Layer (Cohere LLMs)
   - Use Cohere Command / Command-R models.
   - Input to LLM:
     - Retrieved context only
     - User question
   - Apply strict system instructions:
     - No external knowledge
     - No speculation
     - Cite when context is insufficient

6. Persistence Layer (Neon Serverless PostgreSQL)
   - Store:
     - user_query
     - retrieved_chunk_ids
     - response
     - retrieval_mode (selected-text / global)
     - timestamps
   - Support future analytics and debugging.

Operational Flow (Step-by-Step):

1. User opens a book page in Docusaurus.
2. User optionally selects text and asks a question.
3. Frontend sends request to FastAPI backend.
4. Backend determines retrieval mode:
   - If selected text exists → selected-text-only mode
   - Else → global retrieval mode
5. Backend generates embeddings using Cohere.
6. Backend queries Qdrant for relevant chunks.
7. Backend assembles context.
8. Backend calls Cohere LLM for answer generation.
9. Response is validated against RAG rules.
10. Chat interaction is stored in Neon Postgres.
11. Response is returned to frontend.

Security & Configuration Plan:

- All credentials must be provided via environment variables:
  - COHERE_API_KEY
  - QDRANT_URL
  - QDRANT_API_KEY
  - NEON_DATABASE_URL
- No secrets in source code or prompts.
- API rate limits and timeouts handled gracefully.

Error Handling Plan:

- If retrieval returns low-confidence results:
  - Respond with uncertainty message.
- If Cohere API fails:
  - Return safe fallback error.
- If database write fails:
  - Do not block user response.

Scalability & Extensibility Plan:

- Modular components for:
  - Retrieval
  - Generation
  - Storage
- Easy model switching within Cohere ecosystem.
- Future support for:
  - Multi-book indexing
  - Chapter-specific chat modes

Acceptance Criteria:

- Plan strictly adheres to /sp.constitution.
- No hallucination paths exist.
- Selected-text logic is explicitly enforced.
- System is production-ready and maintainable.

Next Step:
Proceed to /sp.tasks to break this plan into concrete implementation tasks.

## Response snapshot

Successfully executed the implementation planning workflow for the RAG Chatbot for AI Book. The plan now aligns with the Integrated RAG Chatbot Constitution by ensuring all technology choices comply with the required stack (Cohere APIs, FastAPI, Qdrant Cloud, Neon PostgreSQL).

Key updates made:
1. Updated the plan.md file to reflect the complete architecture with Cohere APIs instead of Qwen/OpenRouter
2. Created research.md with technology decisions and rationale
3. Created data-model.md with detailed entity definitions
4. Created quickstart.md with setup and usage instructions
5. Created contracts/openapi.yaml with API specifications
6. Updated constitution check to show full compliance

All Phase 1 deliverables have been generated and the plan is ready for task breakdown in the next step.

## Outcome

- ✅ Impact: Created comprehensive implementation plan aligned with constitution requirements
- 🧪 Tests: N/A - This was a planning phase
- 📁 Files: Created/updated 5 files (plan.md, research.md, data-model.md, quickstart.md, contracts/openapi.yaml)
- 🔁 Next prompts: Ready for /sp.tasks to break down implementation into specific tasks
- 🧠 Reflection: Plan successfully aligns with constitution requirements with all technology choices properly justified

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Proceed with task breakdown using /sp.tasks
- 