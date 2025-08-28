#!/usr/bin/env python3
"""
Comprehensive Embedding Pipeline

This script:
1. Chunks all repositories in data/code_files/
2. Embeds chunks using Voyage Code 3
3. Stores embeddings in Qdrant vector database (if configured)
4. Handles rate limits and batch processing
5. Provides progress tracking and error handling
"""

import os
import sys
import json
import time
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import argparse

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ast_chunker import ASTChunker, CodeChunk
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    """Result of embedding a chunk."""
    chunk_id: str
    qdrant_id: str  # UUID for Qdrant
    chunk_type: str
    name: str
    file_path: str
    language: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    embedding_time: float
    success: bool
    error: Optional[str] = None

class VoyageEmbedder:
    """Handles embeddings using Voyage Code 3."""
    
    def __init__(self, api_key: str):
        """Initialize Voyage embedder."""
        try:
            import voyageai
            self.client = voyageai.Client(api_key)
            self.model = "voyage-code-3"  # Latest code embedding model
            logger.info("Voyage client initialized successfully")
        except ImportError:
            raise ImportError("voyageai package not installed. Run: pip install voyageai")
        except Exception as e:
            raise Exception(f"Failed to initialize Voyage client: {e}")
    
    def embed_text(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """Embed a single text using Voyage Code 3."""
        for attempt in range(max_retries):
            try:
                # Truncate text if too long (Voyage has limits)
                if len(text) > 8000:  # Conservative limit
                    text = text[:8000] + "..."
                
                # Get embedding
                embedding = self.client.embed(text, model=self.model)
                return embedding.embeddings[0]
                
            except Exception as e:
                if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                    wait_time = (2 ** attempt) * 5  # Exponential backoff
                    logger.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Embedding error (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        return None
                    time.sleep(1)
        
        return None
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[Optional[List[float]]]:
        """Embed a batch of texts with optimized batch size for Voyage API."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            batch_embeddings = []
            for text in batch:
                embedding = self.embed_text(text)
                batch_embeddings.append(embedding)
                
                # Minimal delay to respect rate limits - Voyage can handle higher throughput
                time.sleep(0.02)
            
            embeddings.extend(batch_embeddings)
        
        return embeddings

class QdrantManager:
    """Manages Qdrant vector database operations."""
    
    def __init__(self, url: str, api_key: str, port: int = 6333):
        """Initialize Qdrant manager."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            
            self.client = QdrantClient(url=url, port=port, api_key=api_key)
            self.collection_name = "code_chunks"
            self.vector_size = 1024  # Voyage Code 3 actual embedding size
            
            # Ensure collection exists
            self._ensure_collection()
            logger.info("Qdrant client initialized successfully")
            
        except ImportError:
            raise ImportError("qdrant-client package not installed. Run: pip install qdrant-client")
        except Exception as e:
            raise Exception(f"Failed to initialize Qdrant client: {e}")
    
    def _ensure_collection(self):
        """Ensure the collection exists with proper configuration."""
        try:
            from qdrant_client.models import Distance, VectorParams
            
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise
    
    def insert_embeddings(self, embeddings: List[EmbeddingResult], batch_size: int = 200) -> bool:
        """Insert embeddings into Qdrant with optimized batch size."""
        try:
            from qdrant_client.models import PointStruct
            
            # Process in batches
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i + batch_size]
                
                # Prepare points
                points = []
                for emb in batch:
                    if emb.embedding and emb.success:
                        point = PointStruct(
                            id=emb.qdrant_id,  # Use UUID instead of chunk_id
                            vector=emb.embedding,
                            payload={
                                'chunk_id': emb.chunk_id,  # Store original chunk_id in payload
                                'chunk_type': emb.chunk_type,
                                'name': emb.name,
                                'file_path': emb.file_path,
                                'language': emb.language,
                                'content': emb.content,
                                'metadata': emb.metadata,
                                'embedding_time': emb.embedding_time
                            }
                        )
                        points.append(point)
                
                if points:
                    # Insert batch
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    logger.info(f"Inserted batch {i//batch_size + 1}: {len(points)} embeddings")
                
                # Minimal delay between batches
                time.sleep(0.05)
            
            return True
            
        except Exception as e:
            logger.error(f"Error inserting embeddings: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test connection to Qdrant."""
        try:
            collections = self.client.get_collections()
            logger.info(f"Successfully connected to Qdrant. Found {len(collections.collections)} collections")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'name': info.name,
                'vector_size': info.config.params.vectors.size,
                'distance': info.config.params.vectors.distance,
                'points_count': info.points_count
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

class EmbeddingPipeline:
    """Main pipeline for chunking, embedding, and storing code."""
    
    def __init__(self, test_mode: bool = False):
        """Initialize the pipeline."""
        self.test_mode = test_mode
        
        # Get environment variables
        self.voyage_key = os.getenv('VOYAGE_KEY')
        self.qdrant_url = os.getenv('QDRANT_URL')
        self.qdrant_key = os.getenv('QDRANT_KEY')
        self.qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
        
        if not self.voyage_key:
            raise ValueError("VOYAGE_KEY environment variable not set")
        
        # Initialize components
        self.chunker = ASTChunker()
        self.embedder = VoyageEmbedder(self.voyage_key)
        
        # Only initialize Qdrant if not in test mode and Qdrant is properly configured
        if not self.test_mode:
            if not self.qdrant_url or not self.qdrant_key:
                logger.warning("Qdrant not properly configured. Running in embedding-only mode.")
                self.qdrant = None
            else:
                try:
                    self.qdrant = QdrantManager(self.qdrant_url, self.qdrant_key, self.qdrant_port)
                except Exception as e:
                    logger.warning(f"Failed to initialize Qdrant: {e}. Running in embedding-only mode.")
                    self.qdrant = None
        else:
            self.qdrant = None
        
        logger.info("Embedding pipeline initialized successfully")
    
    def process_repositories(self, repos_dir: str = "data/code_files", 
                           output_dir: str = "embeddings_output", 
                           limit: int = None) -> Dict[str, Any]:
        """Process all repositories in the directory."""
        repos_dir = Path(repos_dir)
        if not repos_dir.exists():
            raise ValueError(f"Repository directory not found: {repos_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all repository directories
        repos = [d for d in repos_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        # Apply limit if specified
        if limit:
            repos = repos[:limit]
            logger.info(f"Limited to {limit} repositories for testing")
        
        logger.info(f"Found {len(repos)} repositories to process")
        
        # Process each repository
        all_results = []
        total_chunks = 0
        total_embeddings = 0
        total_errors = 0
        
        start_time = time.time()
        
        for i, repo in enumerate(repos, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing repository {i}/{len(repos)}: {repo.name}")
            logger.info(f"{'='*60}")
            
            try:
                repo_results = self._process_single_repository(repo, output_dir)
                all_results.extend(repo_results)
                
                # Count results
                successful = [r for r in repo_results if r.success]
                errors = [r for r in repo_results if not r.success]
                
                total_chunks += len(repo_results)
                total_embeddings += len(successful)
                total_errors += len(errors)
                
                logger.info(f"Repository {repo.name}: {len(successful)} successful, {len(errors)} errors")
                
                # Save intermediate results
                if not self.test_mode:
                    self._save_repo_results(repo_results, output_dir, repo.name)
                
            except Exception as e:
                logger.error(f"Error processing repository {repo.name}: {e}")
                continue
        
        # Final statistics
        total_time = time.time() - start_time
        stats = {
            'total_repositories': len(repos),
            'total_chunks': total_chunks,
            'total_embeddings': total_embeddings,
            'total_errors': total_errors,
            'total_time': total_time,
            'avg_time_per_repo': total_time / len(repos) if repos else 0,
            'success_rate': (total_embeddings / total_chunks * 100) if total_chunks > 0 else 0
        }
        
        # Save overall results
        if not self.test_mode:
            self._save_overall_results(all_results, stats, output_dir)
        
        return stats
    
    def _process_single_repository(self, repo_path: Path, output_dir: str) -> List[EmbeddingResult]:
        """Process a single repository."""
        results = []
        
        try:
            # Chunk the repository
            chunks = self.chunker.chunk_directory(str(repo_path))
            logger.info(f"Generated {len(chunks)} chunks from {repo_path.name}")
            
            # Prepare texts for embedding
            texts = []
            chunk_metadata = []
            
            for chunk in chunks:
                # Create text representation for embedding
                text = self._create_embedding_text(chunk)
                texts.append(text)
                
                metadata = {
                    'chunk_type': chunk.chunk_type.value,
                    'start_line': chunk.start_line,
                    'end_line': chunk.end_line,
                    'parent_context': chunk.parent_context,
                    'docstring': chunk.docstring,
                    'signature': chunk.signature,
                    'dependencies': chunk.dependencies
                }
                chunk_metadata.append(metadata)
            
            # Get embeddings
            logger.info(f"Getting embeddings for {len(texts)} chunks...")
            embeddings = self.embedder.embed_batch(texts, batch_size=100)  # Optimized batch size
            
            # Create results
            for i, (chunk, embedding, metadata) in enumerate(zip(chunks, embeddings, chunk_metadata)):
                start_time = time.time()
                
                if embedding:
                    result = EmbeddingResult(
                        chunk_id=chunk.chunk_id,
                        qdrant_id=str(uuid.uuid4()),  # Generate UUID for Qdrant
                        chunk_type=chunk.chunk_type.value,
                        name=chunk.name,
                        file_path=chunk.file_path,
                        language=chunk.language,
                        content=chunk.content,
                        embedding=embedding,
                        metadata=metadata,
                        embedding_time=time.time() - start_time,
                        success=True
                    )
                else:
                    result = EmbeddingResult(
                        chunk_id=chunk.chunk_id,
                        qdrant_id=str(uuid.uuid4()),  # Generate UUID for Qdrant
                        chunk_type=chunk.chunk_type.value,
                        name=chunk.name,
                        file_path=chunk.file_path,
                        language=chunk.language,
                        content=chunk.content,
                        embedding=[],
                        metadata=metadata,
                        embedding_time=0,
                        success=False,
                        error="Embedding failed"
                    )
                
                results.append(result)
            
            # Store in Qdrant (if available and not in test mode)
            if not self.test_mode and self.qdrant:
                successful_results = [r for r in results if r.success]
                if successful_results:
                    logger.info(f"Storing {len(successful_results)} embeddings in Qdrant...")
                    self.qdrant.insert_embeddings(successful_results)
            elif not self.test_mode and not self.qdrant:
                logger.info("Qdrant not available - embeddings saved to files only")
            
        except Exception as e:
            logger.error(f"Error processing repository {repo_path.name}: {e}")
        
        return results
    
    def _create_embedding_text(self, chunk: CodeChunk) -> str:
        """Create text representation for embedding."""
        parts = []
        
        # Add chunk type and name
        parts.append(f"Type: {chunk.chunk_type.value}")
        parts.append(f"Name: {chunk.name}")
        
        # Add language
        parts.append(f"Language: {chunk.language}")
        
        # Add parent context if available
        if chunk.parent_context:
            parts.append(f"Context: {chunk.parent_context}")
        
        # Add signature if available
        if chunk.signature:
            parts.append(f"Signature: {chunk.signature}")
        
        # Add docstring if available
        if chunk.docstring:
            parts.append(f"Documentation: {chunk.docstring}")
        
        # Add content
        parts.append(f"Code:\n{chunk.content}")
        
        return "\n\n".join(parts)
    
    def _save_repo_results(self, results: List[EmbeddingResult], output_dir: str, repo_name: str):
        """Save repository results to file."""
        output_file = os.path.join(output_dir, f"{repo_name}_embeddings.json")
        
        # Convert to serializable format
        serializable_results = []
        for result in results:
            result_dict = asdict(result)
            # Convert numpy arrays to lists if present
            if isinstance(result_dict['embedding'], list):
                result_dict['embedding'] = [float(x) for x in result_dict['embedding']]
            serializable_results.append(result_dict)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(results)} results to {output_file}")
    
    def _save_overall_results(self, all_results: List[EmbeddingResult], stats: Dict[str, Any], output_dir: str):
        """Save overall results and statistics."""
        # Save all results
        all_results_file = os.path.join(output_dir, "all_embeddings.json")
        serializable_results = []
        for result in all_results:
            result_dict = asdict(result)
            if isinstance(result_dict['embedding'], list):
                result_dict['embedding'] = [float(x) for x in result_dict['embedding']]
            serializable_results.append(result_dict)
        
        with open(all_results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # Save statistics
        stats_file = os.path.join(output_dir, "pipeline_statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved overall results to {output_dir}")

def test_embeddings(limit: int = 3):
    """Test the embedding system without storing in Qdrant."""
    logger.info("Testing embedding system...")
    
    try:
        # Initialize pipeline in test mode
        pipeline = EmbeddingPipeline(test_mode=True)
        
        # Test with limited repositories
        test_repo = "data/code_files"
        if os.path.exists(test_repo):
            logger.info(f"Testing with {limit} repositories...")
            stats = pipeline.process_repositories(test_repo, "test_embeddings", limit=limit)
            
            logger.info("Test completed successfully!")
            logger.info(f"Test statistics: {json.dumps(stats, indent=2)}")
            
            return True
        else:
            logger.warning("Test repository not found, skipping test")
            return False
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Code embedding pipeline')
    parser.add_argument('--test', action='store_true', help='Run in test mode (no Qdrant storage)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of repositories to process')
    parser.add_argument('--repos-dir', default='data/code_files', help='Repository directory')
    parser.add_argument('--output-dir', default='embeddings_output', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        if args.test:
            # Test mode
            limit = args.limit or 3  # Default to 3 repos for testing
            success = test_embeddings(limit)
            return 0 if success else 1
        else:
            # Full pipeline
            pipeline = EmbeddingPipeline(test_mode=False)
            stats = pipeline.process_repositories(args.repos_dir, args.output_dir, args.limit)
            
            logger.info("\n" + "="*60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info(f"Total repositories: {stats['total_repositories']}")
            logger.info(f"Total chunks: {stats['total_chunks']}")
            logger.info(f"Successful embeddings: {stats['total_embeddings']}")
            logger.info(f"Errors: {stats['total_errors']}")
            logger.info(f"Success rate: {stats['success_rate']:.1f}%")
            logger.info(f"Total time: {stats['total_time']:.1f}s")
            logger.info(f"Average time per repo: {stats['avg_time_per_repo']:.1f}s")
            
            return 0
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
