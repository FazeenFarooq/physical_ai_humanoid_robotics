"""
Document processing module for the RAG Chatbot.
This module handles the processing and ingestion of textbook content into the vector database.
"""
import asyncio
import logging
from typing import List
from pathlib import Path
import hashlib
from rag_service import DocumentChunk, rag_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles processing of textbook documents for the RAG system."""
    
    def __init__(self):
        self.chunk_size = 512  # Number of words per chunk
        self.overlap = 50      # Number of words to overlap between chunks

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
        
        return chunks

    async def process_pdf(self, file_path: str, metadata: dict = None) -> List[DocumentChunk]:
        """Process a PDF file and convert it to document chunks."""
        try:
            import PyPDF2
        except ImportError:
            logger.error("PyPDF2 not installed. Install with: pip install pypdf")
            return []
        
        if metadata is None:
            metadata = {}
        
        chunks = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                
                # Chunk the page text
                page_chunks = self.chunk_text(text)
                
                for i, chunk in enumerate(page_chunks):
                    # Create a unique ID for this chunk
                    chunk_id = hashlib.md5(f"{file_path}_page{page_num}_chunk{i}_{chunk[:50]}".encode()).hexdigest()
                    
                    # Add page-specific metadata
                    chunk_metadata = metadata.copy()
                    chunk_metadata['page'] = page_num + 1
                    chunk_metadata['source_file'] = file_path
                    chunk_metadata['chunk_index'] = i
                    
                    # Embed the chunk
                    embedding = await rag_service.embed_text(chunk)
                    
                    chunks.append(DocumentChunk(
                        id=chunk_id,
                        content=chunk,
                        embedding=embedding,
                        metadata=chunk_metadata
                    ))
        
        return chunks

    async def process_textbook_content(self, content: str, metadata: dict = None) -> List[DocumentChunk]:
        """Process raw text content and convert it to document chunks."""
        if metadata is None:
            metadata = {}
        
        chunks = []
        text_chunks = self.chunk_text(content)
        
        for i, chunk in enumerate(text_chunks):
            # Create a unique ID for this chunk
            chunk_id = hashlib.md5(f"textbook_content_chunk{i}_{chunk[:50]}".encode()).hexdigest()
            
            # Add metadata
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = i
            chunk_metadata['source'] = 'textbook_content'
            
            # Embed the chunk
            embedding = await rag_service.embed_text(chunk)
            
            chunks.append(DocumentChunk(
                id=chunk_id,
                content=chunk,
                embedding=embedding,
                metadata=chunk_metadata
            ))
        
        return chunks

    async def process_document_file(self, file_path: str, metadata: dict = None) -> bool:
        """Process a document file and store it in the RAG system."""
        try:
            file_path_obj = Path(file_path)
            file_ext = file_path_obj.suffix.lower()
            
            if metadata is None:
                metadata = {'source_file': str(file_path_obj.name)}
            
            logger.info(f"Processing document: {file_path}")
            
            if file_ext == '.pdf':
                chunks = await self.process_pdf(file_path, metadata)
            else:
                # For text files, read content directly
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                chunks = await self.process_textbook_content(content, metadata)
            
            logger.info(f"Processed {len(chunks)} chunks from {file_path}")
            
            # Store chunks in the vector database
            await rag_service.store_document_chunks(chunks)
            
            logger.info(f"Successfully stored {len(chunks)} chunks in vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return False

    async def process_textbook_directory(self, directory_path: str):
        """Process all documents in a directory for the textbook."""
        directory = Path(directory_path)
        
        # Supported file extensions
        supported_ext = ['.pdf', '.txt', '.md', '.docx']
        
        files_to_process = []
        for ext in supported_ext:
            files_to_process.extend(directory.glob(f"*{ext}"))
            files_to_process.extend(directory.glob(f"**/*{ext}"))  # Also check subdirectories
        
        logger.info(f"Found {len(files_to_process)} files to process")
        
        for file_path in files_to_process:
            # Add basic metadata
            metadata = {
                'source_directory': str(directory),
                'file_name': file_path.name,
                'file_path': str(file_path)
            }
            
            success = await self.process_document_file(str(file_path), metadata)
            if not success:
                logger.error(f"Failed to process {file_path}")


async def main():
    """Main function to demonstrate document processing."""
    # Initialize the RAG service first
    await rag_service.initialize()
    
    # Create document processor
    processor = DocumentProcessor()
    
    # Example: Process a single document
    # This would typically be called with the path to your textbook content
    example_text = """
    Chapter 3: Neural Networks and Deep Learning
    
    Neural networks are a fundamental concept in artificial intelligence and machine learning.
    They are computing systems inspired by the human brain's biological neural networks.
    
    The basic building block of a neural network is the artificial neuron or node.
    Each neuron receives inputs, processes them with a weighted sum, applies an activation function,
    and produces an output.
    
    A neural network consists of layers: an input layer, one or more hidden layers,
    and an output layer. The connections between neurons have associated weights that
    are adjusted during the learning process.
    
    Deep learning refers to neural networks with multiple hidden layers.
    These deep architectures can learn complex patterns and representations
    from raw data without needing manual feature engineering.
    
    Backpropagation is the key algorithm used to train neural networks.
    It calculates the gradient of the loss function with respect to each weight
    by the chain rule, and adjusts the weights to minimize the loss.
    """
    
    # Process the example content
    chunks = await processor.process_textbook_content(
        example_text,
        metadata={'chapter': '3', 'topic': 'Neural Networks', 'source': 'example'}
    )
    
    logger.info(f"Created {len(chunks)} chunks from example content")
    
    # Store the chunks in the vector database
    await rag_service.store_document_chunks(chunks)
    
    logger.info("Example content successfully stored in RAG system")


if __name__ == "__main__":
    asyncio.run(main())