#!/usr/bin/env python3
"""
Simple Web Frontend for Semantic Code Search

A Flask-based web interface for searching across the code database with agentic search capabilities.
"""

from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from agentic_search import AgenticSearcher

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize the agentic code searcher
try:
    searcher = AgenticSearcher()
    SEARCHER_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not initialize agentic searcher: {e}")
    SEARCHER_AVAILABLE = False

@app.route('/')
def index():
    """Main search page."""
    stats = {}
    if SEARCHER_AVAILABLE:
        try:
            stats = searcher.get_database_stats()
        except:
            pass
    
    return render_template('index.html', stats=stats, searcher_available=SEARCHER_AVAILABLE)

@app.route('/generate_queries', methods=['POST'])
def generate_queries():
    """Generate search queries without performing searches."""
    if not SEARCHER_AVAILABLE:
        return jsonify({'error': 'Search service not available'}), 500
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        language = data.get('language', '').strip()
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Generate queries only (with language context if specified)
        generated_queries = searcher.generate_queries_only(query, language if language else None)
        
        return jsonify({
            'success': True,
            'original_query': query,
            'generated_queries': generated_queries
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    """Perform agentic semantic search."""
    if not SEARCHER_AVAILABLE:
        return jsonify({'error': 'Search service not available'}), 500
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        limit = int(data.get('limit', 10))
        score_threshold = float(data.get('score_threshold', 0.7))
        language = data.get('language', '').strip()
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Perform agentic search
        if language:
            search_result = searcher.search_by_language(query, language, limit, score_threshold)
        else:
            search_result = searcher.search(query, limit, score_threshold)
        
        # Convert results to serializable format
        search_results = []
        
        # Create a mapping of chunk_id to query for accurate source tracking
        chunk_to_query = {}
        for query_info in search_result.generated_queries:
            if 'results' in query_info and isinstance(query_info['results'], list):
                for result in query_info['results']:
                    if hasattr(result, 'chunk_id'):
                        chunk_to_query[result.chunk_id] = query_info['query']
        
        for result in search_result.combined_results:
            # Get the source query from our mapping
            source_query = chunk_to_query.get(result.chunk_id, "Unknown")
            
            # Simplify file path to show just repo/filename
            file_path = result.file_path
            if file_path and '/' in file_path:
                # Extract repo and filename from path like "data/code_files/repo_name/file.ext"
                path_parts = file_path.split('/')
                if len(path_parts) >= 3 and path_parts[0] == 'data' and path_parts[1] == 'code_files':
                    # Show as "repo_name/file.ext"
                    file_path = '/'.join(path_parts[2:])
            
            search_results.append({
                'score': round(result.score, 3),
                'chunk_type': result.chunk_type,
                'name': result.name,
                'file_path': file_path,
                'language': result.language,
                'content': result.content,
                'metadata': result.metadata,
                'source_query': source_query,
                'frequency': getattr(result, 'frequency', 1),
                'source_queries': getattr(result, 'source_queries', [source_query])
            })
        
        # Convert generated queries to serializable format
        generated_queries = []
        for query_info in search_result.generated_queries:
            # Debug logging to see what's in the query_info
            print(f"DEBUG: Query info - query: {query_info.get('query')}, languages: {query_info.get('languages')}, priority: {query_info.get('priority')}")
            
            generated_queries.append({
                'query': query_info['query'],
                'priority': query_info['priority'],
                'languages': query_info.get('languages', 'all_coding_languages'),
                'results_count': query_info['results_count']
            })
        
        return jsonify({
            'success': True,
            'original_query': search_result.original_query,
            'generated_queries': generated_queries,
            'results': search_results,
            'total_results': search_result.total_results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def stats():
    """Get database statistics."""
    if not SEARCHER_AVAILABLE:
        return jsonify({'error': 'Search service not available'}), 500
    
    try:
        stats = searcher.get_database_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
