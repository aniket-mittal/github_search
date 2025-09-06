#!/usr/bin/env python3
"""
Semantic Code Search

This script performs semantic search across the code database using:
- Voyage Code 3 for query embedding
- Qdrant vector database with cosine similarity
- Rich metadata display for search results
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Result of a semantic search."""
    score: float
    chunk_id: str
    chunk_type: str
    name: str
    file_path: str
    language: str
    content: str
    metadata: Dict[str, Any]
    frequency: int = 1  # Number of queries that found this result
    source_queries: List[str] = None  # List of queries that found this result

class CodeSearcher:
    """Performs semantic search across the code database."""
    
    def __init__(self):
        """Initialize the code searcher."""
        self.voyage_key = os.getenv('VOYAGE_KEY')
        self.qdrant_url = os.getenv('QDRANT_URL')
        self.qdrant_key = os.getenv('QDRANT_KEY')
        self.qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
        
        if not all([self.voyage_key, self.qdrant_url, self.qdrant_key]):
            raise ValueError("Missing required environment variables: VOYAGE_KEY, QDRANT_URL, QDRANT_KEY")
        
        # Initialize components
        self._init_voyage()
        self._init_qdrant()
        
        logger.info("Code searcher initialized successfully")
    
    def _init_voyage(self):
        """Initialize Voyage embedder."""
        try:
            import voyageai
            self.voyage_client = voyageai.Client(self.voyage_key)
            self.model = "voyage-code-3"
            logger.info("Voyage client initialized successfully")
        except ImportError:
            raise ImportError("voyageai package not installed. Run: pip install voyageai")
        except Exception as e:
            raise Exception(f"Failed to initialize Voyage client: {e}")
    
    def _init_qdrant(self):
        """Initialize Qdrant client with timeout configuration."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as rest
            
            # Initialize with timeout configuration
            self.qdrant_client = QdrantClient(
                url=self.qdrant_url, 
                port=self.qdrant_port, 
                api_key=self.qdrant_key,
                timeout=60.0,  # 60 second timeout for operations
                prefer_grpc=False  # Use HTTP for better timeout handling
            )
            
            # Test connection with a simple operation
            try:
                collections = self.qdrant_client.get_collections()
                logger.info(f"Connected to Qdrant successfully. Found {len(collections.collections)} collections")
            except Exception as conn_error:
                logger.warning(f"Initial connection test failed: {conn_error}")
                logger.info("Continuing with client initialization...")
            
            logger.info("Qdrant client initialized successfully")
        except ImportError:
            raise ImportError("qdrant-client package not installed. Run: pip install qdrant-client")
        except Exception as e:
            raise Exception(f"Failed to initialize Qdrant client: {e}")
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a search query using Voyage Code 3."""
        try:
            # Truncate query if too long
            if len(query) > 8000:
                query = query[:8000] + "..."
            
            embedding = self.voyage_client.embed(query, model=self.model)
            return embedding.embeddings[0]
            
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise
    
    def search(self, query: str, limit: int = 10, score_threshold: float = 0.7) -> List[SearchResult]:
        """Perform semantic search across the code database with retry logic."""
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Embed the query
                logger.info(f"Embedding query: {query}")
                query_vector = self.embed_query(query)
                
                # Search in Qdrant
                logger.info(f"Searching in Qdrant (attempt {attempt + 1}/{max_retries})...")
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                
                # Perform vector search with timeout handling and performance optimizations
                search_results = self.qdrant_client.search(
                    collection_name="code_chunks",
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                    timeout=120,  # Increased timeout for optimized search (2 minutes)
                    search_params={
                        "hnsw_ef": 128,        # Higher ef for better accuracy + speed
                        "exact": False          # Approximate search for speed
                    }
                )
                
                # Convert to SearchResult objects
                results = []
                for result in search_results:
                    payload = result.payload
                    search_result = SearchResult(
                        score=result.score,
                        chunk_id=payload.get('chunk_id', ''),
                        chunk_type=payload.get('chunk_type', ''),
                        name=payload.get('name', ''),
                        file_path=payload.get('file_path', ''),
                        language=payload.get('language', ''),
                        content=payload.get('content', ''),
                        metadata=payload.get('metadata', {})
                    )
                    results.append(search_result)
                
                logger.info(f"Found {len(results)} results")
                return results
                
            except Exception as e:
                error_msg = str(e).lower()
                if "timeout" in error_msg or "timed out" in error_msg:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Qdrant search timeout (attempt {attempt + 1}/{max_retries}). Retrying in {delay}s...")
                        import time
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Qdrant search failed after {max_retries} attempts due to timeout")
                        raise Exception(f"Search operation timed out after {max_retries} attempts. Please try again.")
                else:
                    logger.error(f"Error during search: {e}")
                    raise
        
        # This should never be reached, but just in case
        raise Exception("Search operation failed unexpectedly")
    
    def search_by_language(self, query: str, language: str, limit: int = 10, score_threshold: float = 0.7) -> List[SearchResult]:
        """Search for code in a specific programming language with retry logic."""
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Embed the query
                query_vector = self.embed_query(query)
                
                # Create filter for specific language
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                
                language_filter = Filter(
                    must=[
                        FieldCondition(
                            key="language",
                            match=MatchValue(value=language)
                        )
                    ]
                )
                
                # Search with language filter and timeout handling with performance optimizations
                search_results = self.qdrant_client.search(
                    collection_name="code_chunks",
                    query_vector=query_vector,
                    query_filter=language_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                    timeout=120,  # Increased timeout for optimized search (2 minutes)
                    search_params={
                        "hnsw_ef": 128,        # Higher ef for better accuracy + speed
                        "exact": False          # Approximate search for speed
                    }
                )
                
                # Convert to SearchResult objects
                results = []
                for result in search_results:
                    payload = result.payload
                    search_result = SearchResult(
                        score=result.score,
                        chunk_id=payload.get('chunk_id', ''),
                        chunk_type=payload.get('chunk_type', ''),
                        name=payload.get('name', ''),
                        file_path=payload.get('file_path', ''),
                        language=payload.get('language', ''),
                        content=payload.get('content', ''),
                        metadata=payload.get('metadata', {})
                    )
                    results.append(search_result)
                
                return results
                
            except Exception as e:
                error_msg = str(e).lower()
                if "timeout" in error_msg or "timed out" in error_msg:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Qdrant language search timeout (attempt {attempt + 1}/{max_retries}). Retrying in {delay}s...")
                        import time
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Qdrant language search failed after {max_retries} attempts due to timeout")
                        raise Exception(f"Language search operation timed out after {max_retries} attempts. Please try again.")
                else:
                    logger.error(f"Error during language-specific search: {e}")
                    raise
        
        # This should never be reached, but just in case
        raise Exception("Language search operation failed unexpectedly")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the code database."""
        try:
            collection_info = self.qdrant_client.get_collection("code_chunks")
            return {
                'total_points': collection_info.points_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance': collection_info.config.params.vectors.distance
            }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the Qdrant connection."""
        try:
            # Test basic connectivity
            collections = self.qdrant_client.get_collections()
            
            # Test collection access
            collection_info = self.qdrant_client.get_collection("code_chunks")
            
            return {
                'status': 'healthy',
                'collections_count': len(collections.collections),
                'code_chunks_points': collection_info.points_count,
                'connection_url': f"{self.qdrant_url}:{self.qdrant_port}"
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'connection_url': f"{self.qdrant_url}:{self.qdrant_port}"
            }

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Semantic Code Search')
    parser.add_argument('query', help='Search query')
    parser.add_argument('--limit', type=int, default=10, help='Maximum number of results')
    parser.add_argument('--score-threshold', type=float, default=0.7, help='Minimum similarity score')
    parser.add_argument('--language', help='Filter by programming language')
    
    args = parser.parse_args()
    
    try:
        searcher = CodeSearcher()
        
        # Get database stats
        stats = searcher.get_database_stats()
        print(f"\n📊 Database Statistics:")
        print(f"   Total code chunks: {stats.get('total_points', 'Unknown')}")
        print(f"   Vector dimensions: {stats.get('vector_size', 'Unknown')}")
        print(f"   Similarity metric: {stats.get('distance', 'Unknown')}")
        
        # Perform search
        if args.language:
            print(f"\n🔍 Searching for '{args.query}' in {args.language} code...")
            results = searcher.search_by_language(
                args.query, args.language, args.limit, args.score_threshold
            )
        else:
            print(f"\n🔍 Searching for '{args.query}'...")
            results = searcher.search(args.query, args.limit, args.score_threshold)
        
        # Display results
        if results:
            print(f"\n✅ Found {len(results)} results:")
            print("=" * 80)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result.score:.3f}")
                print(f"   Type: {result.chunk_type}")
                print(f"   Name: {result.name}")
                print(f"   Language: {result.language}")
                print(f"   File: {result.file_path}")
                print(f"   Content Preview: {result.content[:200]}...")
                print("-" * 40)
        else:
            print("\n❌ No results found. Try adjusting the score threshold or query.")
            
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
