# AI Textbook with RAG Chatbot

An advanced AI textbook implementation with retrieval-augmented generation (RAG) capabilities. This project combines a Docusaurus frontend with a FastAPI backend to deliver an intelligent chatbot that can answer questions about the textbook content.

## Project Structure

- `backend/` - FastAPI backend with RAG functionality, vector database integration, and chat history
- `frontend/` - Docusaurus-based website with integrated chat interface (ChatKit)
- `.env.example` - Example environment variables file

## Security Notice

This project handles API keys and sensitive information securely using environment variables. **Never commit API keys or sensitive data to version control.**

## Installation

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables by copying the example file:
   ```bash
   cp .env.example .env
   ```

4. Edit `.env` with your API keys and configuration

### Frontend Setup
1. Install JavaScript dependencies:
   ```bash
   yarn
   ```

## Local Development

### Backend Development
```bash
cd backend
python -m uvicorn main:app --reload
```

### Frontend Development
```bash
yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Environment Variables

Create a `.env` file in the respective directories with the following variables:

Backend (in `backend/.env`):
```bash
# Qdrant Vector Database Configuration
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key

# OpenRouter API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key

# Database Configuration
DATABASE_URL=postgresql://user:password@host:port/database

# Application Settings
DEBUG=false
FRONTEND_URL=http://localhost:3000
```

## Build

### Backend Build
```bash
cd backend
# Make sure environment variables are set
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend Build
```bash
yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

Using SSH:

```bash
USE_SSH=true yarn deploy
```

Not using SSH:

```bash
GIT_USER=<Your GitHub username> yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
