#!/usr/bin/env python3
"""
Mini-Graph Batch Processor
==========================

This processor creates individual mini-graphs per repository instead of 
a single large database. This approach provides:

1. Better memory management - no large edge copying
2. Faster processing - parallel repository processing
3. Better scalability - each repo is independent
4. Easier debugging - isolated failures don't affect other repos
5. Faster queries - smaller database per repo
"""

import os
import sys
import logging
import time
import argparse
import signal
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import json
import multiprocessing

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cpg_core import CPGBuilder
from cpg_processor import CPGBatchProcessor
from mini_graph_store import MiniGraphManager, MiniGraphStore, MiniGraphConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/logs/mini_graph_processing.log')
    ]
)
logger = logging.getLogger(__name__)


class MiniGraphBatchProcessor:
    """Batch processor that creates mini-graphs per repository."""
    
    def __init__(self, mini_graph_manager: MiniGraphManager, max_workers: int = 4,
                 max_file_size: int = 10 * 1024 * 1024):
        """
        Initialize the mini-graph batch processor.
        
        Args:
            mini_graph_manager: Mini-graph manager instance
            max_workers: Maximum number of worker threads
            max_file_size: Maximum file size to process (in bytes)
        """
        self.mini_graph_manager = mini_graph_manager
        self.max_workers = max_workers
        self.max_file_size = max_file_size
        self.cpg_builder = CPGBuilder()
        self.cpg_processor = CPGBatchProcessor(
            max_workers=1,  # Use single worker for CPG generation to avoid memory issues
            chunk_size=10,  # Smaller chunks for memory efficiency
            max_file_size=max_file_size
        )
        
        # Progress tracking
        self.processed_repos = 0
        self.failed_repos = 0
        self.total_repos = 0
        self.start_time = None
        self.should_stop = threading.Event()
        
        # Statistics
        self.stats = {
            'repositories_processed': 0,
            'repositories_failed': 0,
            'total_files_processed': 0,
            'total_nodes_stored': 0,
            'total_edges_stored': 0,
            'processing_time': 0.0,
            'errors': []
        }
        
        logger.info(f"Initialized MiniGraph Batch Processor with {max_workers} workers")
    
    def discover_repositories(self, code_files_dir: str) -> List[str]:
        """
        Discover all repository directories in code_files.
        
        Args:
            code_files_dir: Path to code_files directory
            
        Returns:
            List of repository directory paths
        """
        code_files_path = Path(code_files_dir)
        
        if not code_files_path.exists():
            logger.error(f"Code files directory does not exist: {code_files_dir}")
            return []
        
        # Get all subdirectories (repositories)
        repositories = []
        for item in code_files_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                repositories.append(str(item))
        
        repositories.sort()
        logger.info(f"Discovered {len(repositories)} repositories in {code_files_dir}")
        
        return repositories
    
    def process_repository(self, repo_path: str) -> Dict[str, Any]:
        """
        Process a single repository and store its CPG in a mini-graph.
        
        Args:
            repo_path: Path to the repository directory
            
        Returns:
            Dictionary with processing results
        """
        repo_name = Path(repo_path).name
        start_time = time.time()
        
        try:
            logger.info(f"Processing repository: {repo_name}")
            
            # Check if we should stop processing
            if self.should_stop.is_set():
                return {
                    'repository': repo_name,
                    'success': False,
                    'error': 'Processing stopped by user',
                    'files_processed': 0,
                    'nodes_stored': 0,
                    'edges_stored': 0,
                    'processing_time': 0.0
                }
            
            # Get mini-graph store for this repository
            store = self.mini_graph_manager.get_store(repo_name)
            
            # Check if already processed (by checking if database exists and has data)
            try:
                existing_stats = store.get_repository_stats()
                if existing_stats['total_nodes'] > 0:
                    logger.info(f"Repository {repo_name} already processed, skipping")
                    return {
                        'repository': repo_name,
                        'success': True,
                        'error': None,
                        'files_processed': existing_stats.get('file_count', 0),
                        'nodes_stored': existing_stats['total_nodes'],
                        'edges_stored': existing_stats['total_edges'],
                        'processing_time': 0.0,
                        'skipped': True
                    }
            except Exception:
                # Database doesn't exist or has issues, proceed with processing
                pass
            
            # Discover files in repository
            files = self.cpg_processor.discover_files(
                repo_path,
                exclude_patterns=[
                    '*/.git/*',
                    '*/node_modules/*',
                    '*/venv/*',
                    '*/__pycache__/*',
                    '*/build/*',
                    '*/dist/*',
                    '*/target/*',
                    '*test*',
                    '*Test*',
                    '*.min.js',
                    '*.min.css'
                ]
            )
            
            if not files:
                logger.warning(f"No supported files found in repository: {repo_name}")
                return {
                    'repository': repo_name,
                    'success': True,
                    'error': None,
                    'files_processed': 0,
                    'nodes_stored': 0,
                    'edges_stored': 0,
                    'processing_time': time.time() - start_time
                }
            
            logger.info(f"Found {len(files)} files to process in {repo_name}")
            
            # Create combined CPG for entire repository
            combined_cpg = self._create_repository_cpg(files, repo_path, repo_name)
            
            if not combined_cpg.nodes:
                logger.warning(f"No CPG nodes generated for repository: {repo_name}")
                return {
                    'repository': repo_name,
                    'success': True,
                    'error': None,
                    'files_processed': len(files),
                    'nodes_stored': 0,
                    'edges_stored': 0,
                    'processing_time': time.time() - start_time
                }
            
            # Store the combined CPG using streaming approach
            commit_hash = self._get_commit_hash(repo_path)
            success = store.store_cpg_streaming(combined_cpg, commit_hash)
            
            if not success:
                raise Exception("Failed to store CPG in mini-graph")
            
            # Get final statistics
            final_stats = store.get_repository_stats()
            processing_time = time.time() - start_time
            
            logger.info(f"Completed repository {repo_name}: {len(files)} files, "
                       f"{final_stats['total_nodes']} nodes, {final_stats['total_edges']} edges "
                       f"in {processing_time:.2f}s")
            
            return {
                'repository': repo_name,
                'success': True,
                'error': None,
                'files_processed': len(files),
                'nodes_stored': final_stats['total_nodes'],
                'edges_stored': final_stats['total_edges'],
                'processing_time': processing_time
            }
        
        except Exception as e:
            error_msg = f"Failed to process repository {repo_name}: {str(e)}"
            logger.error(error_msg)
            
            return {
                'repository': repo_name,
                'success': False,
                'error': error_msg,
                'files_processed': 0,
                'nodes_stored': 0,
                'edges_stored': 0,
                'processing_time': time.time() - start_time
            }
    
    def _create_repository_cpg(self, files: List[str], repo_path: str, repo_name: str):
        """Create a combined CPG for the entire repository."""
        from cpg_core import CodePropertyGraph
        
        combined_cpg = CodePropertyGraph()
        combined_cpg.metadata = {
            'repository_name': repo_name,
            'repository_path': repo_path,
            'total_files': len(files),
            'languages': set(),
            'processing_start': time.time()
        }
        
        files_processed = 0
        
        # Process files in smaller batches to manage memory
        batch_size = 20
        for i in range(0, len(files), batch_size):
            if self.should_stop.is_set():
                break
            
            file_batch = files[i:i + batch_size]
            
            for file_path in file_batch:
                if self.should_stop.is_set():
                    break
                
                try:
                    # Generate CPG for single file
                    file_cpg = self.cpg_builder.build_cpg(file_path)
                    
                    if file_cpg.nodes:
                        # Merge file CPG into combined CPG efficiently
                        self._merge_cpg_streaming(combined_cpg, file_cpg)
                        
                        # Track language
                        if file_cpg.metadata.get('language'):
                            combined_cpg.metadata['languages'].add(file_cpg.metadata['language'])
                    
                    files_processed += 1
                    
                    # Log progress for every 20 files
                    if files_processed % 20 == 0:
                        logger.info(f"Processed {files_processed}/{len(files)} files in {repo_name}")
                
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue
        
        # Convert languages set to list for JSON serialization
        combined_cpg.metadata['languages'] = list(combined_cpg.metadata['languages'])
        combined_cpg.metadata['files_processed'] = files_processed
        combined_cpg.metadata['processing_end'] = time.time()
        
        return combined_cpg
    
    def _merge_cpg_streaming(self, target_cpg: 'CodePropertyGraph', source_cpg: 'CodePropertyGraph'):
        """Merge source CPG into target CPG using memory-efficient streaming."""
        # Add nodes - direct reference, no copying
        for node_id, node in source_cpg.nodes.items():
            if node_id not in target_cpg.nodes:
                target_cpg.nodes[node_id] = node
        
        # Add edges - use iterator to avoid copying
        for edge_id, edge in source_cpg.edges.items():
            if edge_id not in target_cpg.edges:
                target_cpg.edges[edge_id] = edge
    
    def _get_commit_hash(self, repo_path: str) -> str:
        """
        Get commit hash for repository if it's a git repository.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Commit hash or empty string
        """
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        
        return ''
    
    def process_repositories_batch(self, repositories: List[str], 
                                  progress_callback: callable = None) -> Dict[str, Any]:
        """
        Process a batch of repositories.
        
        Args:
            repositories: List of repository paths to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with processing statistics
        """
        self.total_repos = len(repositories)
        self.start_time = time.time()
        
        logger.info(f"Starting batch processing of {self.total_repos} repositories")
        
        # Process repositories using thread pool
        results = []
        
        if self.max_workers == 1:
            # Single-threaded processing
            for repo_path in repositories:
                if self.should_stop.is_set():
                    break
                
                result = self.process_repository(repo_path)
                results.append(result)
                self._update_stats(result)
                
                if progress_callback:
                    progress_callback(len(results), self.total_repos)
        else:
            # Multi-threaded processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_repo = {
                    executor.submit(self.process_repository, repo_path): repo_path
                    for repo_path in repositories
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_repo):
                    if self.should_stop.is_set():
                        # Cancel remaining futures
                        for f in future_to_repo:
                            f.cancel()
                        break
                    
                    result = future.result()
                    results.append(result)
                    self._update_stats(result)
                    
                    if progress_callback:
                        progress_callback(len(results), self.total_repos)
        
        # Calculate final statistics
        self.stats['processing_time'] = time.time() - self.start_time
        
        self._log_final_stats()
        
        return self.stats
    
    def _update_stats(self, result: Dict[str, Any]):
        """Update processing statistics with a single result."""
        if result['success']:
            self.stats['repositories_processed'] += 1
            self.stats['total_files_processed'] += result['files_processed']
            self.stats['total_nodes_stored'] += result['nodes_stored']
            self.stats['total_edges_stored'] += result['edges_stored']
        else:
            self.stats['repositories_failed'] += 1
            if result['error']:
                self.stats['errors'].append(result['error'])
    
    def _log_final_stats(self):
        """Log final processing statistics."""
        logger.info("=" * 80)
        logger.info("MINI-GRAPH PROCESSING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total repositories: {self.total_repos}")
        logger.info(f"Successfully processed: {self.stats['repositories_processed']}")
        logger.info(f"Failed: {self.stats['repositories_failed']}")
        logger.info(f"Total files processed: {self.stats['total_files_processed']}")
        logger.info(f"Total nodes stored: {self.stats['total_nodes_stored']:,}")
        logger.info(f"Total edges stored: {self.stats['total_edges_stored']:,}")
        logger.info(f"Processing time: {self.stats['processing_time']:.2f} seconds")
        
        if self.stats['processing_time'] > 0:
            logger.info(f"Repositories per minute: {self.stats['repositories_processed'] / (self.stats['processing_time'] / 60):.2f}")
        
        # Get global mini-graph stats
        global_stats = self.mini_graph_manager.get_global_stats()
        logger.info(f"Total mini-graphs created: {global_stats['total_repositories']}")
        logger.info(f"Total storage size: {global_stats['total_size_mb']:.2f} MB")
        
        if self.stats['errors']:
            logger.warning(f"\nFirst 10 errors:")
            for error in self.stats['errors'][:10]:
                logger.warning(f"  {error}")
    
    def stop_processing(self):
        """Stop the processing gracefully."""
        logger.info("Stopping repository processing...")
        self.should_stop.set()
    
    def save_stats(self, output_file: str):
        """Save processing statistics to file."""
        try:
            # Add global mini-graph stats
            global_stats = self.mini_graph_manager.get_global_stats()
            combined_stats = {
                'processing_stats': self.stats,
                'global_mini_graph_stats': global_stats
            }
            
            with open(output_file, 'w') as f:
                json.dump(combined_stats, f, indent=2, default=str)
            logger.info(f"Saved processing statistics to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")


def progress_callback(current: int, total: int):
    """Simple progress callback function."""
    percentage = (current / total) * 100 if total > 0 else 0
    print(f"\rProgress: {current}/{total} ({percentage:.1f}%)", end='', flush=True)


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print("\nReceived interrupt signal. Stopping processing gracefully...")
    if hasattr(signal_handler, 'processor'):
        signal_handler.processor.stop_processing()


def main():
    """Main entry point for mini-graph batch processor."""
    parser = argparse.ArgumentParser(
        description='Process repositories and create individual mini-graphs'
    )
    parser.add_argument('--code-files-dir', 
                       default='/Users/aniketmittal/Desktop/code/github_search/data/code_files',
                       help='Path to code_files directory')
    parser.add_argument('--storage-dir', default='data/mini_graphs',
                       help='Directory to store mini-graphs')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of worker threads')
    parser.add_argument('--max-file-size', type=int, default=10*1024*1024,
                       help='Maximum file size to process (bytes)')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Repository index to start from (for resuming)')
    parser.add_argument('--limit', type=int,
                       help='Maximum number of repositories to process')
    parser.add_argument('--test-run', action='store_true',
                       help='Process only first 5 repositories for testing')
    parser.add_argument('--stats-file',
                       help='File to save processing statistics')
    parser.add_argument('--clear-repo',
                       help='Clear specific repository data before processing')
    parser.add_argument('--list-repos', action='store_true',
                       help='List existing repositories with mini-graphs')
    parser.add_argument('--global-stats', action='store_true',
                       help='Show global statistics across all mini-graphs')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize mini-graph manager
        config = MiniGraphConfig(storage_dir=args.storage_dir)
        manager = MiniGraphManager(config)
        
        # Handle utility commands
        if args.list_repos:
            repos = manager.list_repositories()
            print(f"Found {len(repos)} repositories with mini-graphs:")
            for repo in repos:
                print(f"  - {repo}")
            return 0
        
        if args.global_stats:
            stats = manager.get_global_stats()
            print("\nGlobal Mini-Graph Statistics:")
            print(f"  Total repositories: {stats['total_repositories']}")
            print(f"  Total nodes: {stats['total_nodes']:,}")
            print(f"  Total edges: {stats['total_edges']:,}")
            print(f"  Total storage: {stats['total_size_mb']:.2f} MB")
            print(f"  Languages: {', '.join(stats['languages'])}")
            return 0
        
        # Clear specific repository if requested
        if args.clear_repo:
            logger.info(f"Clearing repository data: {args.clear_repo}")
            store = manager.get_store(args.clear_repo)
            store.clear_repository()
            return 0
        
        # Initialize processor
        processor = MiniGraphBatchProcessor(
            mini_graph_manager=manager,
            max_workers=args.max_workers,
            max_file_size=args.max_file_size
        )
        
        # Setup signal handler for graceful shutdown
        signal_handler.processor = processor
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Discover repositories
        logger.info(f"Discovering repositories in {args.code_files_dir}")
        repositories = processor.discover_repositories(args.code_files_dir)
        
        if not repositories:
            logger.error("No repositories found to process")
            return 1
        
        # Apply start and limit filters
        if args.start_from > 0:
            repositories = repositories[args.start_from:]
            logger.info(f"Starting from repository index {args.start_from}")
        
        if args.test_run:
            repositories = repositories[:5]
            logger.info("Test run: processing only first 5 repositories")
        elif args.limit:
            repositories = repositories[:args.limit]
            logger.info(f"Limited to {args.limit} repositories")
        
        logger.info(f"Will process {len(repositories)} repositories")
        
        # Process repositories
        stats = processor.process_repositories_batch(
            repositories,
            progress_callback=progress_callback if not args.verbose else None
        )
        
        print()  # New line after progress
        
        # Save statistics if requested
        if args.stats_file:
            processor.save_stats(args.stats_file)
        
        # Return appropriate exit code
        return 0 if stats['repositories_failed'] == 0 else 1
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())