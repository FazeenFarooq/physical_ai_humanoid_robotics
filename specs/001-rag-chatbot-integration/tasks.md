---

description: "Task list for RAG Chatbot Integration in AI Book"
---

# Tasks: RAG Chatbot for AI Book

**Input**: Design documents from `/specs/001-rag-chatbot-integration/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are included per functional requirements

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions


- **Web app**: `backend/src/`, `frontend/src/`
- Paths shown below assume web app structure - adjust based on plan.md structure

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create backend project structure following plan.md in backend/
- [ ] T002 [P] Initialize Python project with FastAPI dependencies in backend/requirements.txt
- [ ] T003 [P] Configure linting and formatting tools (pylint, black, mypy) in backend/

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

Foundational tasks based on the project:

- [ ] T004 Setup Cohere Python SDK for embedding and LLM integration in backend/
- [ ] T005 [P] Configure Qdrant Cloud Free Tier vector store connection in backend/config.py
- [ ] T006 [P] Configure Cohere Command/Command-R models for text generation in backend/config.py
- [ ] T007 Configure FastAPI backend with async Python setup in backend/main.py
- [ ] T008 Setup Neon Serverless PostgreSQL connection in backend/config.py
- [ ] T009 Configure Docusaurus integration with custom backend for chatbot widget

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Ask Questions About Book Content (Priority: P1) 🎯 MVP

**Goal**: Implement core functionality allowing users to ask questions about book content and receive accurate answers sourced from textbook material only.

**Independent Test**: Can be fully tested by asking a question about specific content in the book and verifying that the response is accurate and sourced from the book material.

### Tests for User Story 1 (OPTIONAL - included per functional requirements) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T010 [P] [US1] Contract test for POST /query endpoint in backend/tests/contract/test_query_contract.py
- [ ] T011 [P] [US1] Integration test for question-answering journey in backend/tests/integration/test_query_flow.py

### Implementation for User Story 1

- [ ] T012 [P] [US1] Create ChatSession model in backend/src/models/chat_session.py
- [ ] T013 [P] [US1] Create Message model in backend/src/models/message.py
- [ ] T014 [US1] Implement ChatSessionService in backend/src/services/chat_session_service.py
- [ ] T015 [US1] Implement MessageService in backend/src/services/message_service.py
- [ ] T016 [US1] Implement basic RAGService in backend/src/services/rag_service.py
- [ ] T017 [US1] Implement GET /health endpoint in backend/src/api/v1/health_router.py
- [ ] T018 [US1] Implement POST /query endpoint in backend/src/api/v1/query_router.py
- [ ] T019 [US1] Add validation and error handling for FR-009 in backend/src/api/v1/query_router.py
- [ ] T020 [US1] Add non-hallucination validation for FR-010 in backend/src/services/rag_service.py
- [ ] T021 [US1] Add citation functionality for FR-006 in backend/src/services/rag_service.py

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Specify Text Selection for Context (Priority: P2)

**Goal**: Implement functionality for users to highlight or select specific portions of the textbook content and ask questions based only on that selected text.

**Independent Test**: Can be fully tested by selecting specific text in the book, asking a question related to the selection, and verifying that the response is based only on the selected text.

### Tests for User Story 2 (OPTIONAL - included per functional requirements) ⚠️

- [ ] T022 [P] [US2] Contract test for selected-text context handling in backend/tests/contract/test_selected_text_contract.py
- [ ] T023 [P] [US2] Integration test for selected-text question journey in backend/tests/integration/test_selected_text_flow.py

### Implementation for User Story 2

- [ ] T024 [P] [US2] Create TextSelection model in backend/src/models/text_selection.py
- [ ] T025 [US2] Implement TextSelectionService in backend/src/services/text_selection_service.py
- [ ] T026 [US2] Enhance RAGService with selected-text mode in backend/src/services/rag_service.py
- [ ] T027 [US2] Update POST /query endpoint to detect and use selected text in backend/src/api/v1/query_router.py
- [ ] T028 [US2] Add context priority logic for FR-003 in backend/src/services/rag_service.py
- [ ] T029 [US2] Add retrieval mode tracking for FR-004 in backend/src/models/message.py

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Review Previous Conversations (Priority: P3)

**Goal**: Implement functionality for users to view their previous conversations with the chatbot to reference past questions and answers.

**Independent Test**: Can be fully tested by viewing past conversations after having multiple chats with the system.

### Tests for User Story 3 (OPTIONAL - included per functional requirements) ⚠️

- [ ] T030 [P] [US3] Contract test for conversation history endpoint in backend/tests/contract/test_history_contract.py
- [ ] T031 [P] [US3] Integration test for conversation history retrieval in backend/tests/integration/test_history_flow.py

### Implementation for User Story 3

- [ ] T032 [US3] Implement GET /history endpoint in backend/src/api/v1/history_router.py
- [ ] T033 [US3] Add history retrieval logic in backend/src/services/chat_session_service.py
- [ ] T034 [US3] Add pagination for conversation history in backend/src/api/v1/history_router.py
- [ ] T035 [US3] Enhance session management for history tracking in backend/src/services/session_manager.py

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Data Ingestion Pipeline

**Goal**: Implement the data ingestion pipeline to parse Docusaurus MDX files, chunk content, generate embeddings, and store in Qdrant

- [ ] T036 [P] Create VectorIndex model in backend/src/models/vector_index.py
- [ ] T037 Create MDX parsing utility in backend/src/ingestion/mdx_parser.py
- [ ] T038 Create chunking utility with chapter→section→paragraph hierarchy in backend/src/ingestion/chunker.py
- [ ] T039 Create embedding service for Cohere integration in backend/src/ingestion/embedding_service.py
- [ ] T040 Create Qdrant indexing service in backend/src/ingestion/qdrant_service.py
- [ ] T041 Create run_pipeline.py for complete ingestion workflow in backend/run_pipeline.py
- [ ] T042 Add metadata storage for chapter_id, page_url, chunk_type in backend/src/ingestion/qdrant_service.py

**Checkpoint**: Data ingestion pipeline complete and book content indexed

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T043 [P] Update documentation in docs/
- [ ] T044 Code cleanup and refactoring across services
- [ ] T045 Performance optimization for retrieval and generation
- [ ] T046 [P] Additional unit tests in backend/tests/unit/
- [ ] T047 Groundedness principle compliance check (all responses from book content)
- [ ] T048 Non-hallucination principle compliance check (uncertainty responses when needed)
- [ ] T049 Context priority principle compliance check (selected-text overrides global)
- [ ] T050 Data privacy and logging compliance check
- [ ] T051 Run quickstart.md validation
- [ ] T052 Create README.md with comprehensive documentation
- [ ] T053 Create database schema SQL file in backend/database_schema.sql
- [ ] T054 Add startup scripts (start_server.sh, start_server.bat) in backend/

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Data Ingestion**: Can run in parallel with user stories after foundational
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Integrates with US1 components
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Integrates with US1/US2 components

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together (if tests requested):
- [ ] T010 [P] [US1] Contract test for POST /query endpoint in backend/tests/contract/test_query_contract.py
- [ ] T011 [P] [US1] Integration test for question-answering journey in backend/tests/integration/test_query_flow.py

# Launch all models for User Story 1 together:
- [ ] T012 [P] [US1] Create ChatSession model in backend/src/models/chat_session.py
- [ ] T013 [P] [US1] Create Message model in backend/src/models/message.py
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Add Data Ingestion Pipeline → Index book content
6. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
   - Developer D: Data Ingestion Pipeline
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence