"""
Secure Qdrant Client Test Script
This script demonstrates how to properly connect to Qdrant using environment variables
instead of hardcoding API keys in the source code.
"""

import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve Qdrant configuration from environment variables
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if not qdrant_url or not qdrant_api_key:
    raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables.")

# Initialize Qdrant client with securely loaded credentials
qdrant_client = QdrantClient(
    url=qdrant_url, 
    api_key=qdrant_api_key,
)

# Test the connection
try:
    collections = qdrant_client.get_collections()
    print("Connected to Qdrant successfully!")
    print(f"Available collections: {collections}")
except Exception as e:
    print(f"Error connecting to Qdrant: {e}")