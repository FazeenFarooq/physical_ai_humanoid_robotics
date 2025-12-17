"""
Data Ingestion Pipeline for the RAG Chatbot
Phase I: Source Parsing, Chunking, Embedding, and Indexing
"""
import asyncio
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import re
from bs4 import BeautifulSoup
import markdown

from backend.rag_service import rag_service, DocumentChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata for a document chunk."""
    source_file: str
    chapter: str = ""
    section: str = ""
    page_number: int = 0
    heading: str = ""
    tags: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "source_file": self.source_file,
            "chapter": self.chapter,
            "section": self.section,
            "page_number": self.page_number,
            "heading": self.heading,
            "tags": self.tags or []
        }


class MDXParser:
    """Parses MDX files from Docusaurus to extract content and metadata."""
    
    def __init__(self):
        self.chunk_size = 512  # words per chunk
        self.overlap = 50      # words to overlap between chunks

    def parse_mdx_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse an MDX file and extract content with metadata.
        Returns a list of content parts with metadata.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Extract frontmatter if present
        frontmatter = self._extract_frontmatter(content)
        
        # Remove frontmatter from content
        content_without_frontmatter = self._remove_frontmatter(content)
        
        # Convert MDX/Markdown to plain text while preserving structure
        plain_text = self._mdx_to_text(content_without_frontmatter)
        
        # Extract headings and structure
        headings = self._extract_headings(content_without_frontmatter)
        
        # Create document parts with metadata
        parts = self._create_document_parts(plain_text, headings, frontmatter, file_path)
        
        return parts

    def _extract_frontmatter(self, content: str) -> Dict[str, Any]:
        """Extract YAML frontmatter from the content."""
        import yaml
        
        # Look for frontmatter between --- delimiters
        frontmatter_match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
        if frontmatter_match:
            try:
                frontmatter = yaml.safe_load(frontmatter_match.group(1))
                return frontmatter or {}
            except yaml.YAMLError:
                logger.warning("Failed to parse frontmatter, returning empty dict")
                return {}
        return {}

    def _remove_frontmatter(self, content: str) -> str:
        """Remove frontmatter from content."""
        frontmatter_match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
        if frontmatter_match:
            # Remove the frontmatter but keep the rest
            return content[frontmatter_match.end():].strip()
        return content

    def _mdx_to_text(self, mdx_content: str) -> str:
        """Convert MDX/Markdown to plain text."""
        # Convert to HTML first
        html_content = markdown.markdown(mdx_content)
        
        # Then extract plain text from HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        plain_text = soup.get_text(separator=' ')
        
        # Clean up extra whitespace
        plain_text = re.sub(r'\s+', ' ', plain_text).strip()
        
        return plain_text

    def _extract_headings(self, mdx_content: str) -> List[Dict[str, str]]:
        """Extract headings from MDX content."""
        # Simple regex to find markdown headings
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        lines = mdx_content.split('\n')
        
        headings = []
        for line in lines:
            match = re.match(heading_pattern, line.strip())
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                headings.append({
                    'level': level,
                    'title': title
                })
        
        return headings

    def _create_document_parts(self, text: str, headings: List[Dict], frontmatter: Dict, file_path: str) -> List[Dict[str, Any]]:
        """Create document parts from text and metadata."""
        # Split text into chunks
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Find the most relevant heading for this chunk
            relevant_heading = self._find_relevant_heading(headings, i, self.chunk_size)
            
            # Create metadata
            metadata = DocumentMetadata(
                source_file=file_path,
                chapter=frontmatter.get('chapter', ''),
                section=frontmatter.get('section', ''),
                page_number=frontmatter.get('page_number', 0),
                heading=relevant_heading,
                tags=frontmatter.get('tags', [])
            )
            
            chunks.append({
                'text': chunk_text,
                'metadata': metadata.to_dict()
            })
        
        return chunks

    def _find_relevant_heading(self, headings: List[Dict], position: int, chunk_size: int) -> str:
        """Find the most relevant heading for a given text position."""
        if not headings:
            return ""
        
        # For now, return the last heading before this position
        # This is a simplified approach - in a real implementation, 
        # we might want to use character position instead of word position
        relevant_heading = ""
        for heading in headings:
            # Simplified: just return the last heading
            relevant_heading = heading['title']
        
        return relevant_heading


class DataIngestionPipeline:
    """Main pipeline for ingesting textbook content into the RAG system."""
    
    def __init__(self):
        self.parser = MDXParser()
    
    async def process_document(self, file_path: str) -> List[DocumentChunk]:
        """Process a single document file and convert to chunks."""
        file_path_obj = Path(file_path)
        file_ext = file_path_obj.suffix.lower()
        
        if file_ext in ['.md', '.mdx']:
            # Process as MDX/Markdown file
            document_parts = self.parser.parse_mdx_file(file_path)
        else:
            # For other files, just read as plain text
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Simple chunking without sophisticated metadata
            document_parts = self.parser._create_document_parts(
                content, [], {}, file_path
            )
        
        chunks = []
        for i, part in enumerate(document_parts):
            # Create unique ID for this chunk
            chunk_id = hashlib.md5(
                f"{file_path}_chunk{i}_{part['text'][:50]}".encode()
            ).hexdigest()
            
            # Embed the text
            embedding = await rag_service.embed_text(part['text'])
            
            # Create DocumentChunk
            chunk = DocumentChunk(
                id=chunk_id,
                content=part['text'],
                embedding=embedding,
                metadata=part['metadata']
            )
            
            chunks.append(chunk)
        
        return chunks
    
    async def process_directory(self, directory_path: str):
        """Process all documents in a directory."""
        directory = Path(directory_path)
        
        # Supported file extensions
        supported_ext = ['.md', '.mdx', '.txt', '.pdf']
        
        files_to_process = []
        for ext in supported_ext:
            files_to_process.extend(directory.glob(f"*{ext}"))
            files_to_process.extend(directory.glob(f"**/*{ext}"))  # Subdirectories too
        
        logger.info(f"Found {len(files_to_process)} files to process")
        
        for file_path in files_to_process:
            logger.info(f"Processing {file_path}")
            
            try:
                chunks = await self.process_document(str(file_path))
                logger.info(f"Created {len(chunks)} chunks from {file_path}")
                
                # Store chunks in the vector database
                await rag_service.store_document_chunks(chunks)
                
                logger.info(f"Successfully stored {len(chunks)} chunks for {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
    
    async def run_pipeline(self, source_path: str):
        """Run the complete data ingestion pipeline."""
        logger.info("Starting data ingestion pipeline...")
        
        if Path(source_path).is_file():
            # Process a single file
            chunks = await self.process_document(source_path)
            await rag_service.store_document_chunks(chunks)
            logger.info(f"Processed single file: {source_path}")
        else:
            # Process a directory
            await self.process_directory(source_path)
        
        logger.info("Data ingestion pipeline completed!")


async def main():
    """Main function to demonstrate the ingestion pipeline."""
    # Initialize the RAG service first
    await rag_service.initialize()
    
    # Create the ingestion pipeline
    ingestion_pipeline = DataIngestionPipeline()
    
    # Example: Process Docusaurus docs directory
    docs_path = "../docs"  # Adjust this to your actual docs path
    
    if Path(docs_path).exists():
        await ingestion_pipeline.run_pipeline(docs_path)
    else:
        logger.warning(f"Docs directory not found: {docs_path}")
        # Create a sample document for testing
        sample_content = """
        # Introduction to Neural Networks
        
        Neural networks are a fundamental concept in artificial intelligence and machine learning. They are computing systems inspired by the human brain's biological neural networks.
        
        ## Basic Structure
        
        The basic building block of a neural network is the artificial neuron or node. Each neuron receives inputs, processes them with a weighted sum, applies an activation function, and produces an output.
        
        A neural network consists of layers: an input layer, one or more hidden layers, and an output layer. The connections between neurons have associated weights that are adjusted during the learning process.
        
        ## Deep Learning
        
        Deep learning refers to neural networks with multiple hidden layers. These deep architectures can learn complex patterns and representations from raw data without needing manual feature engineering.
        
        Backpropagation is the key algorithm used to train neural networks. It calculates the gradient of the loss function with respect to each weight by the chain rule, and adjusts the weights to minimize the loss.
        """
        
        # Write sample content to a file for testing
        with open("sample_textbook.md", "w", encoding="utf-8") as f:
            f.write(sample_content)
        
        await ingestion_pipeline.run_pipeline("sample_textbook.md")
        import os
        os.remove("sample_textbook.md")  # Clean up after test


if __name__ == "__main__":
    asyncio.run(main())