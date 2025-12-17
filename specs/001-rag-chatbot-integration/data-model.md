# Data Model: RAG Chatbot for AI Book

## Entities

### ChatSession
Represents a single user's conversation with the chatbot, containing metadata about the session.

**Fields:**
- `id` (UUID): Unique identifier for the session
- `user_id` (UUID, nullable): Identifier for authenticated users
- `created_at` (timestamp): When the session started
- `updated_at` (timestamp): When the session was last active
- `session_token` (string): JWT token for session authentication

**Validation:**
- `id` must be unique
- `created_at` must be before `updated_at`

### Message
An individual exchange in the conversation, including the user's query and the system's response.

**Fields:**
- `id` (UUID): Unique identifier for the message
- `session_id` (UUID): Reference to the chat session
- `role` (string): Either "user" or "assistant"
- `content` (text): The message content
- `created_at` (timestamp): When the message was created
- `retrieved_chunks` (JSON): IDs and metadata of chunks used for response (assistant messages only)
- `retrieval_mode` (string): "selected-text" or "global"
- `sources` (JSON): Citations to specific parts of the textbook

**Validation:**
- `role` must be one of the allowed values
- `retrieval_mode` must be one of the allowed values
- Messages in a session must be chronologically ordered

### TextSelection
A portion of text from the AI book that the user has selected for context.

**Fields:**
- `id` (UUID): Unique identifier for the selection
- `session_id` (UUID): Reference to the chat session
- `content` (text): The selected text content
- `page_url` (string): URL of the page where text was selected
- `chapter_id` (string): ID of the chapter where text was selected
- `created_at` (timestamp): When the selection was made

**Validation:**
- `content` must not exceed 5000 characters
- `page_url` must be a valid URL

### VectorIndex
A representation of the book content transformed into vectors for efficient similarity search.

**Fields:**
- `chunk_id` (string): Unique identifier for the content chunk
- `content` (text): The chunk text content
- `embedding` (vector): The vector representation of the content
- `metadata` (JSON): Additional information including chapter_id, page_url, chunk_type
- `created_at` (timestamp): When the vector was generated

**Validation:**
- `chunk_id` must be unique
- `embedding` must be the correct dimensional vector
- `chunk_type` in metadata must be one of "chapter", "section", or "paragraph"