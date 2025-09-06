#!/usr/bin/env python3
"""
Process Local Repositories Script
=================================

This script processes the repositories stored locally in data/code_files
using the enhanced CPG daemon system.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from batch_daemon import BatchProcessingDaemon, RepositoryJob

# Configure logging to see detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s',
    handlers=[
        logging.FileHandler('local_repo_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def discover_local_repositories() -> List[Dict[str, Any]]:
    """Discover all local repositories in data/code_files."""
    data_dir = Path("/Users/aniketmittal/Desktop/code/github_search/data/code_files")
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return []
    
    repositories = []
    repo_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    logger.info(f"Found {len(repo_dirs)} repository directories")
    
    for i, repo_dir in enumerate(sorted(repo_dirs), 1):  # Process all repositories
        repo_name = repo_dir.name
        
        # Try to detect primary language by file extensions
        language = detect_primary_language(repo_dir)
        
        repo_info = {
            'name': repo_name,
            'path': str(repo_dir),
            'language': language,
            'priority': 0,
            'max_files': None,  # No file limit - process all files
            'local': True
        }
        
        repositories.append(repo_info)
        
        if i % 10 == 0:
            logger.info(f"Cataloged {i} repositories...")
    
    logger.info(f"Cataloged {len(repositories)} repositories total")
    return repositories


def detect_primary_language(repo_path: Path) -> str:
    """Detect primary language of a repository by counting file extensions."""
    language_counts = {
        'python': 0,
        'javascript': 0, 
        'java': 0,
        'go': 0,
        'rust': 0,
        'cpp': 0,
        'c': 0,
        'csharp': 0,
        'php': 0,
        'ruby': 0,
        'swift': 0,
        'kotlin': 0,
        'scala': 0,
        'unknown': 0
    }
    
    extension_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript', 
        '.ts': 'javascript',
        '.tsx': 'javascript',
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.php': 'php',
        '.rb': 'ruby',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.kts': 'kotlin',
        '.scala': 'scala'
    }
    
    try:
        for file_path in repo_path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in extension_map:
                    language_counts[extension_map[ext]] += 1
                else:
                    language_counts['unknown'] += 1
    except Exception as e:
        logger.debug(f"Error analyzing {repo_path}: {e}")
    
    # Return language with most files
    primary_language = max(language_counts, key=language_counts.get)
    return primary_language if language_counts[primary_language] > 0 else 'multi'


class LocalRepositoryProcessor:
    """Modified daemon for processing local repositories."""
    
    def __init__(self):
        # Configure daemon for local processing
        config = {
            'max_workers': 4,  # Increased workers for full processing
            'storage_dir': 'data/mini_graphs',
            'state_file': 'local_processing_state.json',
            'compression': True,
            'batch_size': 500,  # Larger batches for efficiency
            'max_file_size': None,  # No file size limit
            'max_files_per_repo': None,  # No file count limit per repo
            'monitor_interval': 60,  # Monitor every minute
            'checkpoint_interval': 600,  # Save state every 10 minutes
        }
        
        self.daemon = BatchProcessingDaemon(config)
        logger.info("LocalRepositoryProcessor initialized")
    
    def add_local_repository(self, repo_info: Dict[str, Any]) -> str:
        """Add a local repository to processing queue."""
        # Create a custom job for local processing
        job = RepositoryJob(
            name=repo_info['name'],
            url=repo_info['path'],  # Use local path instead of URL
            language=repo_info['language'],
            priority=repo_info.get('priority', 0),
            max_files=repo_info.get('max_files', None),
            exclude_patterns=['test', 'tests', '.git', 'node_modules', '__pycache__']
        )
        
        # Add metadata to identify as local repo
        if job.stats is None:
            job.stats = {}
        job.stats['local_repo'] = True
        
        self.daemon.job_queue.put(job)
        self.daemon.stats.jobs_total += 1
        
        return job.name
    
    def process_repositories(self, repo_list: List[Dict[str, Any]]):
        """Process a list of local repositories."""
        logger.info(f"Starting to process {len(repo_list)} local repositories")
        
        # Add all repositories to queue
        for repo_info in repo_list:
            job_name = self.add_local_repository(repo_info)
            logger.info(f"Queued repository: {job_name}")
        
        logger.info(f"All {len(repo_list)} repositories queued. Starting daemon...")
        
        # Start daemon processing
        try:
            self.daemon.start()
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        except Exception as e:
            logger.error(f"Processing error: {e}")
        finally:
            logger.info("Processing completed")


def main():
    """Main entry point for local repository processing."""
    logger.info("=" * 60)
    logger.info("STARTING LOCAL REPOSITORY PROCESSING")
    logger.info("=" * 60)
    
    # Discover local repositories
    logger.info("Phase 1: Discovering local repositories...")
    repositories = discover_local_repositories()
    
    if not repositories:
        logger.error("No repositories found to process")
        return 1
    
    logger.info(f"Found {len(repositories)} repositories to process")
    
    # Show sample of what we found
    logger.info("\nSample repositories:")
    for i, repo in enumerate(repositories[:5]):
        logger.info(f"  {i+1}. {repo['name']} ({repo['language']})")
    
    # Initialize processor
    logger.info("\nPhase 2: Initializing processor...")
    processor = LocalRepositoryProcessor()
    
    # Start processing
    logger.info("\nPhase 3: Starting repository processing...")
    logger.info("*** DETAILED OUTPUT MONITORING BEGINS ***")
    processor.process_repositories(repositories)
    
    logger.info("=" * 60)
    logger.info("LOCAL REPOSITORY PROCESSING COMPLETE")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())