#!/usr/bin/env python3
"""
Script to run the complete RAG Chatbot pipeline:

Phase I:   Data Ingestion - Parse Docusaurus MDX files, chunk, embed, and index
Phase II:  Backend Core - Initialize FastAPI, DB, endpoints
Phase III: Testing - Validate the complete workflow
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the backend directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from backend.ingestion import DataIngestionPipeline
from backend.rag_service import rag_service
from backend.config import settings


async def run_ingestion_pipeline():
    """Run the complete data ingestion pipeline."""
    print("🚀 Starting Phase I: Data Ingestion Pipeline...")
    
    # Initialize the RAG service first (needed for embeddings)
    print("  Initializing RAG service...")
    await rag_service.initialize()
    
    # Create the ingestion pipeline
    print("  Creating ingestion pipeline...")
    ingestion_pipeline = DataIngestionPipeline()
    
    # Try to find the Docusaurus docs directory
    docs_path = Path("../docs")  # Relative to backend directory
    content_path = Path("../content")  # Another common location for content
    
    source_path = None
    if docs_path.exists():
        source_path = docs_path
    elif content_path.exists():
        source_path = content_path
    else:
        print(f"  ⚠️  Docs directory not found at {docs_path} or {content_path}")
        print("  Creating sample content for testing...")
        
        # Create sample content
        sample_dir = Path("sample_docs")
        sample_dir.mkdir(exist_ok=True)
        
        # Create a sample textbook chapter
        sample_content = """---
title: Introduction to Neural Networks
chapter: 3
section: 3.1
tags: [ai, neural-networks, machine-learning]
---

# Introduction to Neural Networks

Neural networks are a fundamental concept in artificial intelligence and machine learning. They are computing systems inspired by the human brain's biological neural networks.

## Basic Structure

The basic building block of a neural network is the artificial neuron or node. Each neuron receives inputs, processes them with a weighted sum, applies an activation function, and produces an output.

A neural network consists of layers: an input layer, one or more hidden layers, and an output layer. The connections between neurons have associated weights that are adjusted during the learning process.

## Deep Learning

Deep learning refers to neural networks with multiple hidden layers. These deep architectures can learn complex patterns and representations from raw data without needing manual feature engineering.

Backpropagation is the key algorithm used to train neural networks. It calculates the gradient of the loss function with respect to each weight by the chain rule, and adjusts the weights to minimize the loss.

## Applications

Neural networks have been successfully applied to various domains including:
- Computer vision
- Natural language processing
- Speech recognition
- Autonomous vehicles
"""
        
        sample_file = sample_dir / "neural_networks.mdx"
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        
        source_path = sample_file
        print(f"  Created sample content at {sample_file}")
    
    # Run the ingestion pipeline
    print(f"  Processing content from: {source_path}")
    await ingestion_pipeline.run_pipeline(str(source_path))
    
    print("✅ Phase I: Data Ingestion Pipeline completed!")


async def test_backend_core():
    """Test the backend core functionality."""
    print("\n⚙️  Starting Phase II: Backend Core Validation...")
    
    # Test embedding functionality
    print("  Testing embedding functionality...")
    test_text = "Artificial neural networks are computing systems."
    embedding = await rag_service.embed_text(test_text)
    print(f"  Generated embedding of size: {len(embedding)}")
    
    # Test RAG response with and without selected context
    print("  Testing RAG response without selected context...")
    response1 = await rag_service.process_query(
        query="What are neural networks?",
        context_selection=None
    )
    print(f"  Response: {response1.answer[:100]}...")
    print(f"  Sources: {response1.sources}")
    
    print("  Testing RAG response with selected context...")
    selected_text = "Neural networks are computing systems inspired by the human brain's biological neural networks."
    response2 = await rag_service.process_query(
        query="What are neural networks?",
        context_selection=selected_text
    )
    print(f"  Response: {response2.answer[:100]}...")
    print(f"  Sources: {response2.sources}")
    
    print("✅ Phase II: Backend Core Validation completed!")


async def test_complete_workflow():
    """Test the complete RAG workflow."""
    print("\n🔄 Starting Phase III: Complete Workflow Test...")
    
    # Test query with retrieved context
    print("  Testing query with retrieved context...")
    response = await rag_service.process_query(
        query="What are the main components of a neural network?",
        user_id="test_user_123"
    )
    
    print(f"  Query: What are the main components of a neural network?")
    print(f"  Response: {response.answer}")
    print(f"  Sources: {response.sources}")
    print(f"  Confidence: {response.confidence}")
    print(f"  Processing time: {response.processing_time:.2f}s")
    
    # Test with user-selected context (higher priority)
    print("\n  Testing with user-selected context (higher priority)...")
    selected_context = "The main components are the input layer, hidden layers, and output layer."
    response = await rag_service.process_query(
        query="What are the main components of a neural network?",
        user_id="test_user_123",
        context_selection=selected_context
    )
    
    print(f"  Query: What are the main components of a neural network?")
    print(f"  Selected Context: {selected_context}")
    print(f"  Response: {response.answer}")
    print(f"  Sources: {response.sources}")
    
    print("✅ Phase III: Complete Workflow Test completed!")


async def main():
    """Run the complete RAG Chatbot pipeline."""
    print("🌟 Starting Complete RAG Chatbot Pipeline")
    print(f"   Configuration:")
    print(f"   - Qdrant URL: {settings.qdrant_url or 'Not configured'}")
    print(f"   - OpenRouter Model: {settings.openrouter_default_model}")
    print(f"   - Database: {settings.database_url or 'Not configured'}")
    
    try:
        # Phase I: Run ingestion pipeline
        await run_ingestion_pipeline()
        
        # Phase II: Test backend core
        await test_backend_core()
        
        # Phase III: Test complete workflow
        await test_complete_workflow()
        
        print("\n🎉 All phases completed successfully!")
        print("💡 The RAG Chatbot backend is ready for integration with the Docusaurus frontend.")
        
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())