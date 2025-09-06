#!/usr/bin/env python3
"""
Simple Qdrant vector retrieval tool
"""

import sys
from pathlib import Path

# Add parent directory to import semantic_search
sys.path.append(str(Path(__file__).parent.parent))
from semantic_search import CodeSearcher

def test_connection():
    """Test Qdrant connection and return True if successful."""
    try:
        searcher = CodeSearcher()
        health = searcher.health_check()
        print(f"✅ Connection successful: {health['status']}")
        print(f"   Code chunks: {health['code_chunks_points']}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def get_user_query():
    """Get search query from user input."""
    print("\n🔍 What code would you like to search for?")
    query = input("Enter your search query: ").strip()
    return query if query else "function"

def search_similar_vectors(query, threshold=0.65, limit=1):
    """Search for similar vectors and display top 5 results above threshold."""
    try:
        searcher = CodeSearcher()
        results = searcher.search(query, limit=limit, score_threshold=threshold)
        
        if results and len(results) > 0:
            print(f"\n✅ Found {len(results)} similar vectors:")
            print("=" * 60)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result.score:.3f}")
                print(f"   File: {result.file_path}")
                print(f"   Language: {result.language}")
                print(f"   Content: {result.content[:200]}...")
                print("-" * 40)
            
            # Return the content of the first result
            return results[0].content
        else:
            print(f"\n❌ No results found above threshold {threshold}")
            return "No similar code found for the given query."

    except Exception as e:
        print(f"❌ Search failed: {e}")
        return f"Error searching for code: {str(e)}"

def main():
    """Main function that orchestrates the three operations."""
    print("🚀 Qdrant Vector Retrieval Tool")
    
    # Test connection
    if not test_connection():
        return
    
    # Get user query
    query = get_user_query()
    
    # Search and display results
    search_similar_vectors(query)

if __name__ == "__main__":
    main()
