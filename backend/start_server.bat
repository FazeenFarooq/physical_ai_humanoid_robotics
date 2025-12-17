@echo off
REM Startup script for the RAG Chatbot FastAPI backend

echo Starting RAG Chatbot backend service...

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Warning: venv not found, assuming global packages are installed
)

REM Set environment variables
set PYTHONPATH=%PYTHONPATH%;%~dp0

REM Start the FastAPI application
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

pause