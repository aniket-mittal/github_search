#!/usr/bin/env python3
"""
Agentic Search Module

This module provides an intelligent search layer that breaks down complex user queries
into multiple focused search queries using GPT-4o-mini, then combines and ranks the results.
"""

import os
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from semantic_search import CodeSearcher, SearchResult

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AgenticSearchResult:
    """Result of an agentic search with query breakdown."""
    original_query: str
    generated_queries: List[Dict[str, Any]]
    combined_results: List[SearchResult]
    total_results: int

class AgenticSearcher:
    """Intelligent search that breaks down complex queries using GPT-4o-mini."""
    
    def __init__(self):
        """Initialize the agentic searcher."""
        self.openai_key = os.getenv('OPENAI_KEY')
        if not self.openai_key:
            raise ValueError("Missing required environment variable: OPENAI_KEY")
        
        # Initialize the underlying code searcher
        self.code_searcher = CodeSearcher()
        
        # Initialize OpenAI client
        self._init_openai()
        
        logger.info("Agentic searcher initialized successfully")
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=self.openai_key)
            logger.info("OpenAI client initialized successfully")
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        except Exception as e:
            raise Exception(f"Failed to initialize OpenAI client: {e}")
    
    def _generate_search_queries(self, user_query: str, language: str = None) -> List[Dict[str, Any]]:
        """
        Use GPT-4o-mini to break down a complex user query into focused search queries.
        
        Args:
            user_query: The original user query
            language: Optional programming language to filter by
            
        Returns:
            List of dictionaries containing generated queries with priority and ranking
        """
        try:
            # Define language categories for the LLM
            coding_languages = [
                "python", "javascript", "typescript", "react", "react-ts", "java", "cpp", "c", "csharp",
                "php", "ruby", "go", "rust", "swift", "kotlin", "scala", "clojure", "haskell", "ocaml",
                "fsharp", "r", "matlab", "bash", "sql"
            ]
            
            text_languages = [
                "html", "css", "scss", "sass", "xml", "yaml", "json", "toml", "ini", "config", 
                "markdown", "text"
            ]
            
            # Build language-specific instructions
            language_instructions = ""
            if language:
                # User has selected a specific language, so bypass LLM language selection
                language_instructions = f"""
IMPORTANT: The user has specifically selected to search in {language} code only. 
Focus your search queries on concepts and patterns that are relevant to {language} development.
Do NOT include language-specific terms in your queries - focus on the core concepts and functionality.
"""
            else:
                # Let the LLM choose appropriate language filtering
                language_instructions = f"""
LANGUAGE FILTERING: You can choose to focus on specific programming languages or language types based on the query:

1. For general programming concepts: Use "all_coding_languages" to search across all programming languages
2. For text/documentation: Use "text_languages" to search in markdown, HTML, CSS, config files, etc.
3. For specific language needs: Provide a list of specific languages like ["python", "javascript", "go"]

Available language categories:
- Coding languages: {', '.join(coding_languages)}
- Text languages: {', '.join(text_languages)}

IMPORTANT: Never include language names in your actual search queries. Focus on concepts, patterns, and functionality.
"""

            system_prompt = f"""You are an expert code search assistant. Your job is to break down complex user queries into focused, specific search queries that will help find relevant code.

{language_instructions}

Rules:
1. Generate between 3-15 search queries depending on the complexity
2. Each query should be focused on a specific aspect or concept
3. Queries should be different enough to find diverse results
4. Include both broad and specific queries
5. Keep queries concise and focused
6. Assign priorities carefully: HIGH for core/essential concepts, MEDIUM for important but secondary, LOW for nice-to-have or edge cases
7. Provide a ranking number for each query where 1 is most important and N is least important (N = total number of queries generated)
8. NEVER include programming language names in your search queries - focus on concepts and functionality

Format your response as a JSON array with objects containing:
- "query": the search query string (keep it concise, NO language names)
- "priority": high/medium/low based on how important this aspect is
- "ranking": number from 1 to N where 1 is most important and N is least important
- "languages": language filter (only include if user didn't specify a language):
  * "all_coding_languages" for general programming concepts
  * "text_languages" for documentation/config files
  * ["python", "javascript"] for specific language needs

Example for 5 queries:
[
  {{
    "query": "user authentication login",
    "priority": "high",
    "ranking": 1,
    "languages": "all_coding_languages"
  }},
  {{
    "query": "password hashing bcrypt",
    "priority": "high",
    "ranking": 2,
    "languages": "all_coding_languages"
  }},
  {{
    "query": "session management",
    "priority": "medium",
    "ranking": 3,
    "languages": "all_coding_languages"
  }},
  {{
    "query": "JWT token handling",
    "priority": "medium",
    "ranking": 4,
    "languages": "all_coding_languages"
  }},
  {{
    "query": "error handling auth",
    "priority": "low",
    "ranking": 5,
    "languages": "all_coding_languages"
  }}
]"""

            user_prompt = f"""Break down this search query into focused search queries for finding relevant code:

User Query: "{user_query}"

Generate focused search queries that will help find the most relevant code snippets."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse the response
            content = response.choices[0].message.content
            import json
            try:
                # Try to extract JSON from the response
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_str = content[json_start:json_end].strip()
                elif "```" in content:
                    json_start = content.find("```") + 3
                    json_end = content.find("```", json_start)
                    json_str = content[json_start:json_end].strip()
                else:
                    json_str = content.strip()
                
                queries = json.loads(json_str)
                
                # Log the LLM-generated prompts, priority, language, and rank
                print("\n=== LLM GENERATED QUERIES ===")
                for i, query in enumerate(queries):
                    print(f"Query {i+1}: {query.get('query', 'N/A')}")
                    print(f"  Priority: {query.get('priority', 'N/A')}")
                    print(f"  Languages: {query.get('languages', 'N/A')}")
                    print(f"  Ranking: {query.get('ranking', 'N/A')}")
                    print()
                print("===============================\n")
                
                # Validate the structure
                if not isinstance(queries, list):
                    raise ValueError("Response is not a list")
                
                # Ensure each query has required fields
                validated_queries = []
                for query in queries:
                    if isinstance(query, dict) and "query" in query:
                        # If user specified a language, use that instead of LLM's language choice
                        if language:
                            query_languages = language
                        else:
                            query_languages = query.get("languages", "all_coding_languages")
                        
                        validated_query = {
                            "query": query.get("query", ""),
                            "priority": query.get("priority", "medium"),
                            "ranking": query.get("ranking", 0), # Default to 0 if ranking is missing
                            "languages": query_languages
                        }
                        if validated_query["query"].strip():
                            validated_queries.append(validated_query)
                
                if not validated_queries:
                    # Fallback: create simple queries from the original
                    fallback_languages = language if language else "all_coding_languages"
                    validated_queries = [
                        {"query": user_query, "priority": "high", "ranking": 1, "languages": fallback_languages}
                    ]
                
                return validated_queries
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse OpenAI response as JSON: {e}")
                # Fallback: return the original query
                fallback_languages = language if language else "all_coding_languages"
                return [
                    {"query": user_query, "priority": "high", "ranking": 1, "languages": fallback_languages}
                ]
                
        except Exception as e:
            logger.error(f"Error generating search queries: {e}")
            # Fallback: return the original query
            fallback_languages = language if language else "all_coding_languages"
            return [
                {"query": user_query, "priority": "high", "ranking": 1, "languages": fallback_languages}
            ]
    
    def _combine_and_rank_results(self, all_results: List[SearchResult], limit: int, query_results: List[Dict[str, Any]]) -> List[SearchResult]:
        """
        Combine results from multiple queries and rank them by:
        1. Number of queries that returned this result (cross-query frequency)
        2. Priority (high > medium > low)
        3. Ranking (1-N)
        4. Score
        
        Args:
            all_results: List of all search results from different queries
            limit: Maximum number of results to return
            query_results: List of query information with priority levels and rankings
            
        Returns:
            Ranked and deduplicated results sorted by cross-query frequency, priority, ranking, then score
        """
        if not all_results:
            return []
        
        # Count how many queries each result appears in and track which queries found each result
        chunk_frequency = {}
        chunk_to_queries = {}  # Track which specific queries found each result
        chunk_to_priority = {}
        chunk_to_ranking = {}
        
        for query_info in query_results:
            if 'results' in query_info and isinstance(query_info['results'], list):
                for result in query_info['results']:
                    if hasattr(result, 'chunk_id'):
                        # Count frequency across queries
                        chunk_frequency[result.chunk_id] = chunk_frequency.get(result.chunk_id, 0) + 1
                        
                        # Track which queries found this result
                        if result.chunk_id not in chunk_to_queries:
                            chunk_to_queries[result.chunk_id] = []
                        chunk_to_queries[result.chunk_id].append(query_info['query'])
                        
                        # Track priority and ranking (use highest priority and lowest ranking if multiple)
                        current_priority = query_info['priority']
                        current_ranking = query_info.get('ranking', 0)
                        
                        if result.chunk_id not in chunk_to_priority:
                            chunk_to_priority[result.chunk_id] = current_priority
                            chunk_to_ranking[result.chunk_id] = current_ranking
                        else:
                            # Update priority if current is higher
                            priority_scores = {"high": 3, "medium": 2, "low": 1}
                            if priority_scores.get(current_priority, 0) > priority_scores.get(chunk_to_priority[result.chunk_id], 0):
                                chunk_to_priority[result.chunk_id] = current_priority
                            
                            # Update ranking if current is lower (lower = higher importance)
                            if current_ranking < chunk_to_ranking[result.chunk_id]:
                                chunk_to_ranking[result.chunk_id] = current_ranking
        
        # Remove duplicates based on chunk_id and add priority/frequency info
        seen_chunks = set()
        unique_results = []
        
        for result in all_results:
            if result.chunk_id not in seen_chunks:
                seen_chunks.add(result.chunk_id)
                
                # Get the frequency, priority and ranking from our mapping
                result_frequency = chunk_frequency.get(result.chunk_id, 1)
                result_priority = chunk_to_priority.get(result.chunk_id, "medium")
                result_ranking = chunk_to_ranking.get(result.chunk_id, 0)
                
                # Create a result with frequency, priority and ranking info for sorting
                result_with_info = (result, result_frequency, result_priority, result_ranking)
                unique_results.append(result_with_info)
        
        # Sort by frequency first (higher = better), then priority, then ranking, then score
        def sort_key(item):
            result, frequency, priority, ranking = item
            priority_score = {"high": 3, "medium": 2, "low": 1}[priority]
            # Higher frequency = higher importance, lower ranking number = higher importance
            return (-frequency, -priority_score, ranking, -result.score)
        
        unique_results.sort(key=sort_key)
        
        # Log the sorting for debugging
        logger.info("Sorting results by cross-query frequency, priority, and ranking:")
        for result, frequency, priority, ranking in unique_results[:min(5, len(unique_results))]:
            logger.info(f"  Frequency: {frequency}, {priority.upper()} (rank {ranking}): {result.chunk_id} (score: {result.score:.3f})")
        
        # Extract just the results and update their frequency and source_queries fields
        sorted_results = []
        for result, frequency, _, _ in unique_results:
            # Update the frequency and source_queries fields of the SearchResult object
            result.frequency = frequency
            result.source_queries = chunk_to_queries.get(result.chunk_id, [])
            sorted_results.append(result)
        
        return sorted_results[:limit]
    
    def search(self, query: str, limit: int = 10, score_threshold: float = 0.7) -> AgenticSearchResult:
        """
        Perform agentic search by breaking down the query and combining results.
        
        Args:
            query: The user's search query
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            AgenticSearchResult containing generated queries and combined results
        """
        logger.info(f"Starting agentic search for query: {query}")
        
        # Generate focused search queries
        generated_queries = self._generate_search_queries(query)
        logger.info(f"Generated {len(generated_queries)} search queries")
        
        # Debug: Log the generated queries to see language info
        for i, q in enumerate(generated_queries):
            logger.info(f"Generated query {i+1}: '{q.get('query')}' with languages: {q.get('languages')}")
        
        # Check if the LLM selected a specific language (not "all_coding_languages" or "text_languages")
        selected_language = None
        logger.info("Checking if LLM selected a specific language...")
        
        for i, q in enumerate(generated_queries):
            languages = q.get("languages")
            logger.info(f"Query {i+1} languages field: {languages} (type: {type(languages)})")
            
            if languages and languages not in ["all_coding_languages", "text_languages"]:
                if isinstance(languages, list) and len(languages) == 1:
                    # Single language selected
                    selected_language = languages[0]
                    logger.info(f"Found single language in list: {selected_language}")
                    break
                elif isinstance(languages, str) and languages not in ["all_coding_languages", "text_languages"]:
                    # Single language selected
                    selected_language = languages
                    logger.info(f"Found single language string: {selected_language}")
                    break
        
        if selected_language:
            logger.info(f"LLM selected specific language: {selected_language}, applying language filtering")
            # Use the same logic as search_by_language
            return self._search_with_language_filtering(query, selected_language, limit, score_threshold, generated_queries)
        else:
            logger.info("No specific language selected by LLM, proceeding without language filtering")
        
        # Perform searches for each generated query (no language filtering)
        all_results = []
        query_results = []
        
        for query_info in generated_queries:
            try:
                search_query = query_info["query"]
                logger.info(f"Searching with query: {search_query}")
                
                # Perform the search
                results = self.code_searcher.search(search_query, limit * 2, score_threshold)
                
                # Store results for this query
                query_results.append({
                    "query": search_query,
                    "priority": query_info["priority"],
                    "ranking": query_info.get("ranking", 0),
                    "languages": query_info.get("languages", "all_coding_languages"),  # Preserve the languages field
                    "results_count": len(results),
                    "results": results
                })
                
                # Add to combined results
                all_results.extend(results)
                
            except Exception as e:
                logger.error(f"Error searching with query '{search_query}': {e}")
                continue
        
        # Combine and rank all results
        combined_results = self._combine_and_rank_results(all_results, limit, query_results)
        
        logger.info(f"Combined search returned {len(combined_results)} unique results")
        
        return AgenticSearchResult(
            original_query=query,
            generated_queries=query_results,
            combined_results=combined_results,
            total_results=len(combined_results)
        )
    
    def _search_with_language_filtering(self, query: str, language: str, limit: int, score_threshold: float, generated_queries: List[Dict[str, Any]]) -> AgenticSearchResult:
        """
        Helper method to perform search with language filtering.
        Used by both search() and search_by_language() methods.
        
        Args:
            query: The user's search query
            language: Programming language to filter by
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            generated_queries: Pre-generated search queries
            
        Returns:
            AgenticSearchResult containing generated queries and combined results
        """
        logger.info(f"=== ENTERING _search_with_language_filtering ===")
        logger.info(f"Query: {query}")
        logger.info(f"Language: {language}")
        logger.info(f"Limit: {limit}")
        logger.info(f"Score threshold: {score_threshold}")
        logger.info(f"Number of generated queries: {len(generated_queries)}")
        logger.info(f"Performing search with language filtering for: {language}")
        
        # Perform searches for each generated query WITHOUT language filtering (get all results)
        all_results = []
        query_results = []
        
        for query_info in generated_queries:
            try:
                search_query = query_info["query"]
                logger.info(f"Searching with query: {search_query}")
                
                # Perform the search WITHOUT language filter to get all results
                results = self.code_searcher.search(search_query, limit * 3, score_threshold)
                
                # Filter results by language on the client side (case-insensitive and handle variations)
                def normalize_language(lang):
                    if not lang:
                        return ""
                    return lang.lower().strip()
                
                target_lang = normalize_language(language)
                language_filtered_results = [r for r in results if normalize_language(r.language) == target_lang]
                
                # Log language filtering results for debugging
                logger.info(f"Query '{search_query}': Found {len(results)} total results, {len(language_filtered_results)} in {language}")
                if len(results) > 0:
                    unique_languages = set(r.language for r in results)
                    logger.info(f"Available languages in results: {unique_languages}")
                
                # Store results for this query
                query_results.append({
                    "query": search_query,
                    "priority": query_info["priority"],
                    "ranking": query_info.get("ranking", 0),
                    "languages": language,  # Preserve the specific language
                    "results_count": len(language_filtered_results),
                    "results": language_filtered_results
                })
                
                # Add to combined results
                all_results.extend(language_filtered_results)
                
            except Exception as e:
                logger.error(f"Error searching with query '{search_query}': {e}")
                continue
        
        # Combine and rank all results
        combined_results = self._combine_and_rank_results(all_results, limit, query_results)
        
        logger.info(f"Combined search returned {len(combined_results)} unique results in {language}")
        
        # If no results found in the specified language, log a warning
        if len(combined_results) == 0:
            logger.warning(f"No results found in {language} for query: {query}")
        
        return AgenticSearchResult(
            original_query=query,
            generated_queries=query_results,
            combined_results=combined_results,
            total_results=len(combined_results)
        )
    
    def search_by_language(self, query: str, language: str, limit: int = 10, score_threshold: float = 0.7) -> AgenticSearchResult:
        """
        Perform agentic search with language filtering using client-side filtering.
        
        Args:
            query: The user's search query
            language: Programming language to filter by
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            AgenticSearchResult containing generated queries and combined results
        """
        logger.info(f"Starting agentic search for query: {query} in language: {language}")
        
        # Generate focused search queries with language context
        generated_queries = self._generate_search_queries(query, language)
        logger.info(f"Generated {len(generated_queries)} search queries")
        
        # Debug: Log the generated queries to see language info
        for i, q in enumerate(generated_queries):
            logger.info(f"Generated query {i+1}: '{q.get('query')}' with languages: {q.get('languages')}")
        
        # Use the shared language filtering logic
        return self._search_with_language_filtering(query, language, limit, score_threshold, generated_queries)
    
    def generate_queries_only(self, query: str, language: str = None) -> List[Dict[str, Any]]:
        """
        Generate search queries without performing searches.
        This allows the UI to show queries immediately.
        
        Args:
            query: The user's search query
            language: Optional programming language to filter by
            
        Returns:
            List of generated queries with priority and ranking
        """
        logger.info(f"Generating queries for: {query}" + (f" in language: {language}" if language else ""))
        generated_queries = self._generate_search_queries(query, language)
        logger.info(f"Generated {len(generated_queries)} search queries")
        
        return [{
            "query": q["query"],
            "priority": q["priority"],
            "ranking": q.get("ranking", 0),
            "languages": q.get("languages", "all_coding_languages"),
            "status": "generated"
        } for q in generated_queries]
    
    def search_with_generated_queries(self, query: str, generated_queries: List[Dict[str, Any]], limit: int = 10, score_threshold: float = 0.7) -> AgenticSearchResult:
        """
        Perform searches using pre-generated queries.
        
        Args:
            query: The original user query
            generated_queries: List of queries to search with
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            AgenticSearchResult with search results
        """
        logger.info(f"Starting searches with {len(generated_queries)} generated queries")
        
        # Perform searches for each generated query
        all_results = []
        query_results = []
        
        for query_info in generated_queries:
            try:
                search_query = query_info["query"]
                logger.info(f"Searching with query: {search_query}")
                
                # Perform the search
                results = self.code_searcher.search(search_query, limit * 2, score_threshold)
                
                # Store results for this query
                query_results.append({
                    "query": search_query,
                    "priority": query_info["priority"],
                    "ranking": query_info.get("ranking", 0),
                    "results_count": len(results),
                    "results": results,
                    "status": "completed"
                })
                
                # Add to combined results
                all_results.extend(results)
                
            except Exception as e:
                logger.error(f"Error searching with query '{search_query}': {e}")
                query_results.append({
                    "query": search_query,
                    "priority": query_info["priority"],
                    "ranking": query_info.get("ranking", 0),
                    "results_count": 0,
                    "results": [],
                    "status": "error"
                })
                continue
        
        # Combine and rank all results
        combined_results = self._combine_and_rank_results(all_results, limit, query_results)
        
        logger.info(f"Combined search returned {len(combined_results)} unique results")
        
        return AgenticSearchResult(
            original_query=query,
            generated_queries=query_results,
            combined_results=combined_results,
            total_results=len(combined_results)
        )
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics from the underlying code searcher."""
        return self.code_searcher.get_database_stats()
