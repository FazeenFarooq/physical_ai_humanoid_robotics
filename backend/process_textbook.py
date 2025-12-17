"""
Textbook Processing Script for RAG ChatBot
This script will ingest your textbook content, create embeddings, and store them in Qdrant
"""
import asyncio
import os
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import numpy as np
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def hash_text(text: str) -> str:
    """Generate a unique ID for a text chunk"""
    return hashlib.md5(text.encode()).hexdigest()

class TextbookProcessor:
    def __init__(self):
        # Load Qdrant client with secure credentials
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer("Qwen/qwen-2-7b-instruct")
        
        # Collection name for textbook content
        self.collection_name = "textbook_content"
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Check if collection exists, create if not"""
        try:
            self.qdrant_client.get_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' already exists")
        except:
            print(f"Creating collection '{self.collection_name}'")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
            )
    
    def read_textbook_files(self, docs_dir: str):
        """Read all text/md/mdx files from the documentation directory"""
        docs_path = Path(docs_dir)
        content_chunks = []
        
        # Find all markdown and mdx files
        for file_path in docs_path.rglob("*"):
            if file_path.suffix.lower() in ['.md', '.mdx', '.txt']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Split content into chunks
                        chunks = self._split_into_chunks(content, file_path.name)
                        
                        for i, chunk in enumerate(chunks):
                            # Create metadata for the chunk
                            metadata = {
                                "source_file": file_path.name,
                                "section": f"{file_path.stem}_chunk_{i}",
                                "page_info": ""
                            }
                            
                            content_chunks.append({
                                "text": chunk,
                                "metadata": metadata
                            })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        
        return content_chunks
    
    def _split_into_chunks(self, text: str, source: str, chunk_size: int = 500, overlap: int = 50):
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for a nearby sentence ending
                temp_end = end
                while temp_end < len(text) and temp_end < start + chunk_size + 100:
                    if text[temp_end] in '.!?':
                        end = temp_end + 1
                        break
                    temp_end += 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position forward with overlap
            start = end - overlap if end < len(text) else len(text)
            
            # Prevent infinite loop
            if start >= len(text) or end == start:
                break
        
        return chunks
    
    def embed_and_store(self, content_chunks):
        """Generate embeddings and store in Qdrant"""
        print(f"Processing {len(content_chunks)} content chunks...")
        
        points = []
        for i, chunk_data in enumerate(content_chunks):
            text = chunk_data["text"]
            metadata = chunk_data["metadata"]
            
            # Generate embedding
            embedding = self.embedding_model.encode([text])[0].tolist()
            
            # Create point for Qdrant
            point = models.PointStruct(
                id=hash_text(f"{text[:100]}_{i}"),  # Create a unique ID
                vector=embedding,
                payload={
                    "content": text,
                    "metadata": metadata
                }
            )
            
            points.append(point)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(content_chunks)} chunks...")
        
        # Upload points to Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"Successfully stored {len(points)} chunks in Qdrant collection '{self.collection_name}'")
    
    def process_textbook(self, docs_dir: str):
        """Complete process: read, embed, and store textbook content"""
        print("Starting textbook processing...")
        
        # Read content from docs
        content_chunks = self.read_textbook_files(docs_dir)
        print(f"Extracted {len(content_chunks)} content chunks from textbook")
        
        # Embed and store in Qdrant
        self.embed_and_store(content_chunks)
        
        print("Textbook processing completed successfully!")


class RAGChatBot:
    def __init__(self):
        # Initialize Qdrant client with secure credentials
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer("Qwen/qwen-2-7b-instruct")
        
        # Initialize OpenAI-compatible client for OpenRouter
        from openai import OpenAI
        
        self.llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        self.collection_name = "textbook_content"
    
    def search_similar_content(self, query: str, top_k: int = 5):
        """Search for similar content in the vector database"""
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Search in Qdrant
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        results = []
        for hit in search_result:
            results.append({
                "id": str(hit.id),
                "content": hit.payload["content"],
                "metadata": hit.payload["metadata"],
                "score": hit.score
            })
        
        return results
    
    def generate_response(self, query: str, context: list):
        """Generate response using the LLM with provided context"""
        # Format context for the LLM
        context_str = ""
        
        if context:
            # Retrieved chunks
            retrieved_content = "\n\n".join([
                f"[SOURCE: {item['metadata']['source_file']}]\n{item['content'][:500]}"
                for item in context
            ])
            context_str = f"TEXTBOOK CONTENT FOR CONTEXT:\n{retrieved_content}"
        else:
            context_str = "No specific textbook content found. Answer based on general knowledge."
        
        # Construct the final prompt
        prompt = f"""
        You are an AI assistant for an AI textbook. Answer the user's question based on the provided context from the textbook.
        
        CONTEXT FROM TEXTBOOK:
        {context_str}
        
        User's question: {query}
        
        Provide a clear, concise answer citing which sources you used. When citing sources, reference the source file names in brackets like [SOURCE: ...].
        """
        
        # Call the LLM
        response = self.llm_client.chat.completions.create(
            model="qwen/qwen-2-72b-instruct",  # Using Qwen model via OpenRouter
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def chat(self, query: str, top_k: int = 5):
        """Main chat method that performs search and generation"""
        print("Searching textbook for relevant content...")
        
        # Search for relevant content
        context = self.search_similar_content(query, top_k)
        
        print(f"Found {len(context)} relevant sections from the textbook")
        
        # Generate response
        print("Generating response...")
        response = self.generate_response(query, context)
        
        return {
            "response": response,
            "sources": [item["metadata"]["source_file"] for item in context],
            "retrieved_chunks": len(context)
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG ChatBot for AI Textbook")
    parser.add_argument("--action", choices=["process", "chat"], required=True,
                       help="Action to perform: process textbook or chat")
    parser.add_argument("--docs-dir", type=str, default="../docs",
                       help="Path to textbook documentation directory")
    parser.add_argument("--query", type=str,
                       help="Query to ask the chatbot (required for chat action)")
    
    args = parser.parse_args()
    
    if args.action == "process":
        processor = TextbookProcessor()
        processor.process_textbook(args.docs_dir)
        print("Textbook processing completed!")
    
    elif args.action == "chat":
        if not args.query:
            print("Error: Query is required for chat action")
            exit(1)
        
        chatbot = RAGChatBot()
        result = chatbot.chat(args.query)
        
        print("\n" + "="*50)
        print("RESPONSE:")
        print(result["response"])
        print("\nSOURCES:")
        for source in result["sources"]:
            print(f"- {source}")
        print("="*50)