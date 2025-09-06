#!/usr/bin/env python3
"""
Batch Processing Daemon for CPG Generation
==========================================

A high-performance daemon for batch processing repositories to generate
Code Property Graphs. Features:

1. Multi-threaded processing for performance
2. Memory-efficient streaming processing
3. Progress tracking and resume capability
4. Error handling and recovery
5. Integration with mini-graph storage
6. Repository queue management
7. Performance monitoring and optimization
"""

import os
import sys
import json
import time
import signal
import logging
import threading
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import subprocess
import shutil
import tempfile
from contextlib import contextmanager

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_cpg_core import EnhancedCPGBuilder
from mini_graph_store import MiniGraphStore, MiniGraphConfig
from cpg_core import NodeType, EdgeType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s',
    handlers=[
        logging.FileHandler('batch_daemon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class RepositoryJob:
    """Represents a repository processing job."""
    name: str
    url: str
    language: str
    priority: int = 1  # 1=high, 2=medium, 3=low
    max_files: Optional[int] = None
    exclude_patterns: List[str] = None
    status: str = "pending"  # pending, processing, completed, failed
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    stats: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.exclude_patterns is None:
            self.exclude_patterns = []
        if self.stats is None:
            self.stats = {}


@dataclass 
class ProcessingStats:
    """Statistics for daemon processing."""
    jobs_total: int = 0
    jobs_completed: int = 0 
    jobs_failed: int = 0
    jobs_in_progress: int = 0
    files_processed: int = 0
    nodes_generated: int = 0
    edges_generated: int = 0
    processing_time: float = 0.0
    start_time: float = 0.0
    
    def __post_init__(self):
        if self.start_time == 0.0:
            self.start_time = time.time()


class BatchProcessingDaemon:
    """High-performance batch processing daemon."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the batch processing daemon."""
        self.config = config or self._default_config()
        
        # Core components
        self.cpg_builder = EnhancedCPGBuilder()
        self.graph_config = MiniGraphConfig(
            storage_dir=self.config['storage_dir'],
            compression=self.config.get('compression', True),
            batch_size=self.config.get('batch_size', 500)
        )
        
        # Processing state
        self.job_queue = Queue()
        self.active_jobs = {}
        self.completed_jobs = {}
        self.stats = ProcessingStats()
        self.shutdown_event = threading.Event()
        
        # Threading
        self.max_workers = self.config.get('max_workers', 4)
        self.executor = None
        self.monitor_thread = None
        
        # State persistence
        self.state_file = Path(self.config['state_file'])
        self.load_state()
        
        logger.info(f"BatchProcessingDaemon initialized with {self.max_workers} workers")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the daemon."""
        return {
            'max_workers': 4,
            'storage_dir': 'data/mini_graphs',
            'state_file': 'daemon_state.json',
            'compression': True,
            'batch_size': 500,
            'max_file_size': 2 * 1024 * 1024,  # 2MB per file
            'max_files_per_repo': 1000,
            'monitor_interval': 30,  # seconds
            'checkpoint_interval': 300,  # 5 minutes
            'retry_failed_jobs': True,
            'max_retries': 2
        }
    
    def add_repository(self, name: str, url: str, language: str = 'auto', 
                      priority: int = 1, max_files: Optional[int] = None,
                      exclude_patterns: List[str] = None) -> str:
        """Add a repository to the processing queue."""
        job = RepositoryJob(
            name=name,
            url=url,
            language=language,
            priority=priority,
            max_files=max_files if max_files is not None else self.config['max_files_per_repo'],
            exclude_patterns=exclude_patterns or []
        )
        
        self.job_queue.put(job)
        self.stats.jobs_total += 1
        
        logger.info(f"Added repository job: {name} ({url})")
        return job.name
    
    def add_repositories_from_file(self, file_path: str) -> List[str]:
        """Add multiple repositories from a JSON file."""
        with open(file_path, 'r') as f:
            repos = json.load(f)
        
        job_names = []
        for repo in repos:
            name = self.add_repository(
                name=repo['name'],
                url=repo['url'], 
                language=repo.get('language', 'auto'),
                priority=repo.get('priority', 1),
                max_files=repo.get('max_files'),
                exclude_patterns=repo.get('exclude_patterns', [])
            )
            job_names.append(name)
        
        logger.info(f"Added {len(job_names)} repositories from {file_path}")
        return job_names
    
    def start(self):
        """Start the daemon processing."""
        logger.info("Starting BatchProcessingDaemon...")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Start thread pool executor
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Daemon started successfully")
        
        try:
            self._processing_loop()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the daemon gracefully."""
        logger.info("Stopping BatchProcessingDaemon...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for active jobs to complete (with timeout)
        if self.executor:
            logger.info("Waiting for active jobs to complete...")
            self.executor.shutdown(wait=True, timeout=60)
        
        # Save final state
        self.save_state()
        
        logger.info("Daemon stopped gracefully")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}")
        self.shutdown_event.set()
    
    def _processing_loop(self):
        """Main processing loop.""" 
        logger.info("Starting main processing loop...")
        last_checkpoint = time.time()
        
        while not self.shutdown_event.is_set():
            try:
                # Check for checkpoint save
                if time.time() - last_checkpoint > self.config['checkpoint_interval']:
                    self.save_state()
                    last_checkpoint = time.time()
                
                # Process queued jobs
                if not self.job_queue.empty() and len(self.active_jobs) < self.max_workers:
                    try:
                        job = self.job_queue.get(timeout=1)
                        self._submit_job(job)
                    except Empty:
                        continue
                else:
                    # Wait a bit if queue is empty or workers are busy
                    time.sleep(1)
                
                # Check completed jobs
                self._check_completed_jobs()
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(5)  # Brief pause on error
        
        logger.info("Processing loop terminated")
    
    def _submit_job(self, job: RepositoryJob):
        """Submit a job for processing."""
        job.status = "processing"
        job.started_at = time.time()
        
        future = self.executor.submit(self._process_repository, job)
        self.active_jobs[job.name] = {
            'job': job,
            'future': future,
            'started_at': time.time()
        }
        
        self.stats.jobs_in_progress += 1
        logger.info(f"Submitted job for processing: {job.name}")
    
    def _check_completed_jobs(self):
        """Check for completed jobs and update statistics."""
        completed_job_names = []
        
        for job_name, job_info in self.active_jobs.items():
            future = job_info['future']
            if future.done():
                completed_job_names.append(job_name)
                
                try:
                    result = future.result()
                    job = job_info['job']
                    job.status = "completed"
                    job.completed_at = time.time()
                    job.stats = result
                    
                    self.completed_jobs[job_name] = job
                    self.stats.jobs_completed += 1
                    self.stats.jobs_in_progress -= 1
                    
                    # Update global stats
                    if 'files_processed' in result:
                        self.stats.files_processed += result['files_processed']
                    if 'nodes_generated' in result:
                        self.stats.nodes_generated += result['nodes_generated']
                    if 'edges_generated' in result:
                        self.stats.edges_generated += result['edges_generated']
                    
                    logger.info(f"Job completed successfully: {job_name}")
                    
                except Exception as e:
                    job = job_info['job']
                    job.status = "failed"
                    job.error = str(e)
                    job.completed_at = time.time()
                    
                    self.completed_jobs[job_name] = job
                    self.stats.jobs_failed += 1
                    self.stats.jobs_in_progress -= 1
                    
                    logger.error(f"Job failed: {job_name} - {e}")
        
        # Remove completed jobs from active list
        for job_name in completed_job_names:
            del self.active_jobs[job_name]
    
    def _process_repository(self, job: RepositoryJob) -> Dict[str, Any]:
        """Process a single repository job."""
        logger.info(f"Processing repository: {job.name}")
        
        result = {
            'job_name': job.name,
            'files_discovered': 0,
            'files_processed': 0, 
            'files_successful': 0,
            'files_failed': 0,
            'nodes_generated': 0,
            'edges_generated': 0,
            'processing_time': 0.0,
            'languages_detected': {},
            'errors': []
        }
        
        start_time = time.time()
        temp_dir = None
        
        try:
            # Handle local vs remote repositories
            if hasattr(job.stats, 'get') and job.stats.get('local_repo'):
                # Local repository processing
                temp_dir = job.url  # URL is actually the local path
                logger.info(f"Processing local repository {job.name} at {temp_dir}")
                if not os.path.exists(temp_dir):
                    raise Exception(f"Local repository path does not exist: {temp_dir}")
            else:
                # Clone remote repository
                logger.info(f"Cloning repository {job.name}...")
                temp_dir = self._clone_repository(job)
                if not temp_dir:
                    raise Exception("Failed to clone repository")
            
            # Initialize mini graph store for this repository
            graph_store = MiniGraphStore(job.name, self.graph_config)
            
            # Discover source files
            source_files = self._discover_source_files(temp_dir, job.exclude_patterns)
            result['files_discovered'] = len(source_files)
            
            # Limit files if specified
            if job.max_files and len(source_files) > job.max_files:
                source_files = source_files[:job.max_files]
                logger.info(f"Limited processing to {job.max_files} files for {job.name}")
            
            logger.info(f"Processing {len(source_files)} files for {job.name}")
            
            # Process files in batches
            batch_size = self.config['batch_size']
            for i in range(0, len(source_files), batch_size):
                if self.shutdown_event.is_set():
                    logger.info(f"Shutdown requested, stopping processing of {job.name}")
                    break
                
                batch = source_files[i:i + batch_size]
                batch_result = self._process_file_batch(batch, job, graph_store)
                
                # Update results
                for key in ['files_processed', 'files_successful', 'files_failed', 
                           'nodes_generated', 'edges_generated']:
                    result[key] += batch_result.get(key, 0)
                
                # Merge language detection
                for lang, count in batch_result.get('languages_detected', {}).items():
                    result['languages_detected'][lang] = result['languages_detected'].get(lang, 0) + count
                
                result['errors'].extend(batch_result.get('errors', []))
                
                # Progress update
                progress = (i + len(batch)) / len(source_files) * 100
                logger.info(f"Progress for {job.name}: {progress:.1f}% ({i + len(batch)}/{len(source_files)} files)")
            
            # Finalize graph store
            graph_store.close()
            
        except Exception as e:
            logger.error(f"Error processing repository {job.name}: {e}")
            result['errors'].append(str(e))
            raise
        
        finally:
            # Cleanup (only for cloned repositories, not local ones)
            if temp_dir and os.path.exists(temp_dir) and not (hasattr(job.stats, 'get') and job.stats.get('local_repo')):
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
        
        result['processing_time'] = time.time() - start_time
        logger.info(f"Completed processing {job.name} in {result['processing_time']:.1f}s")
        
        return result
    
    def _clone_repository(self, job: RepositoryJob) -> Optional[str]:
        """Clone a repository to temporary directory."""
        try:
            temp_dir = tempfile.mkdtemp(prefix=f"cpg_{job.name}_")
            
            cmd = ['git', 'clone', '--depth=1', job.url, temp_dir]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                return temp_dir
            else:
                logger.error(f"Git clone failed for {job.name}: {result.stderr}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"Git clone timeout for {job.name}")
            return None
        except Exception as e:
            logger.error(f"Git clone error for {job.name}: {e}")
            return None
    
    def _discover_source_files(self, repo_dir: str, exclude_patterns: List[str]) -> List[str]:
        """Discover source files in repository."""
        source_files = []
        
        # Supported extensions
        extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.go', '.rs', 
            '.cpp', '.cc', '.cxx', '.c', '.h', '.hpp', '.cs', '.php', 
            '.rb', '.swift', '.kt', '.scala'
        }
        
        # Default exclude patterns
        default_excludes = {
            '.git', '.svn', 'node_modules', '__pycache__', 'venv', 'env',
            'build', 'dist', 'target', 'bin', 'obj', '.idea', '.vscode',
            'vendor', 'third_party', 'external', 'deps', 'packages', 'test', 'tests'
        }
        
        all_excludes = default_excludes.union(set(exclude_patterns))
        
        for root, dirs, files in os.walk(repo_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in all_excludes]
            
            for file in files:
                if Path(file).suffix.lower() in extensions:
                    file_path = os.path.join(root, file)
                    
                    # Check file size (if limit is set)
                    try:
                        file_size = os.path.getsize(file_path)
                        if self.config['max_file_size'] is None or file_size <= self.config['max_file_size']:
                            source_files.append(file_path)
                    except OSError:
                        continue
        
        return sorted(source_files)
    
    def _process_file_batch(self, files: List[str], job: RepositoryJob, 
                           graph_store: MiniGraphStore) -> Dict[str, Any]:
        """Process a batch of files.""" 
        result = {
            'files_processed': 0,
            'files_successful': 0,
            'files_failed': 0,
            'nodes_generated': 0,
            'edges_generated': 0,
            'languages_detected': {},
            'errors': []
        }
        
        for file_path in files:
            try:
                result['files_processed'] += 1
                
                # Generate CPG
                cpg = self.cpg_builder.build_cpg(file_path)
                
                if cpg.nodes:
                    result['files_successful'] += 1
                    result['nodes_generated'] += len(cpg.nodes)
                    result['edges_generated'] += len(cpg.edges)
                    
                    # Track language
                    language = cpg.metadata.get('language', 'unknown')
                    result['languages_detected'][language] = result['languages_detected'].get(language, 0) + 1
                    
                    # Store in mini graph
                    success = graph_store.store_cpg_streaming(cpg)
                    if not success:
                        logger.warning(f"Failed to store CPG for {file_path}")
                    
                else:
                    result['files_failed'] += 1
                    error = cpg.metadata.get('error', 'No nodes generated')
                    result['errors'].append(f"{Path(file_path).name}: {error}")
            
            except Exception as e:
                result['files_failed'] += 1
                result['errors'].append(f"{Path(file_path).name}: {str(e)}")
        
        return result
    
    def _monitor_loop(self):
        """Background monitoring and statistics loop."""
        while not self.shutdown_event.is_set():
            try:
                self._print_status()
                time.sleep(self.config['monitor_interval'])
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
    
    def _print_status(self):
        """Print current processing status."""
        uptime = time.time() - self.stats.start_time
        
        logger.info("=" * 60)
        logger.info("BATCH PROCESSING DAEMON STATUS")
        logger.info("=" * 60)
        logger.info(f"Uptime: {uptime/3600:.1f} hours")
        logger.info(f"Jobs - Total: {self.stats.jobs_total}, "
                   f"Completed: {self.stats.jobs_completed}, "
                   f"Failed: {self.stats.jobs_failed}, "
                   f"In Progress: {self.stats.jobs_in_progress}")
        logger.info(f"Queue Size: {self.job_queue.qsize()}")
        logger.info(f"Files Processed: {self.stats.files_processed:,}")
        logger.info(f"Nodes Generated: {self.stats.nodes_generated:,}")
        logger.info(f"Edges Generated: {self.stats.edges_generated:,}")
        
        if self.stats.files_processed > 0 and uptime > 0:
            files_per_hour = (self.stats.files_processed / uptime) * 3600
            logger.info(f"Processing Rate: {files_per_hour:.1f} files/hour")
        
        # Active job details
        if self.active_jobs:
            logger.info(f"Active Jobs ({len(self.active_jobs)}):")
            for name, info in self.active_jobs.items():
                elapsed = time.time() - info['started_at']
                logger.info(f"  - {name}: running for {elapsed/60:.1f} minutes")
        
        logger.info("=" * 60)
    
    def save_state(self):
        """Save daemon state to file."""
        try:
            state = {
                'stats': asdict(self.stats),
                'completed_jobs': {name: asdict(job) for name, job in self.completed_jobs.items()},
                'config': self.config,
                'timestamp': time.time()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.debug(f"State saved to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def load_state(self):
        """Load daemon state from file."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                # Restore statistics (but reset current session counters)
                if 'stats' in state:
                    saved_stats = state['stats']
                    self.stats.files_processed = saved_stats.get('files_processed', 0)
                    self.stats.nodes_generated = saved_stats.get('nodes_generated', 0) 
                    self.stats.edges_generated = saved_stats.get('edges_generated', 0)
                
                # Restore completed jobs
                if 'completed_jobs' in state:
                    for name, job_data in state['completed_jobs'].items():
                        job = RepositoryJob(**job_data)
                        self.completed_jobs[name] = job
                
                logger.info(f"State loaded from {self.state_file}")
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current daemon status."""
        return {
            'stats': asdict(self.stats),
            'queue_size': self.job_queue.qsize(),
            'active_jobs': len(self.active_jobs),
            'completed_jobs': len(self.completed_jobs),
            'uptime': time.time() - self.stats.start_time
        }


def main():
    """Main entry point for the batch processing daemon."""
    parser = argparse.ArgumentParser(description='CPG Batch Processing Daemon')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--repos', type=str, help='Repository list JSON file')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads')
    parser.add_argument('--storage-dir', type=str, default='data/mini_graphs', 
                       help='Storage directory for graphs')
    parser.add_argument('--dry-run', action='store_true', help='Validate configuration without running')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'max_workers': args.workers,
        'storage_dir': args.storage_dir
    }
    
    if args.config:
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Initialize daemon
    daemon = BatchProcessingDaemon(config)
    
    # Add repositories if provided
    if args.repos:
        daemon.add_repositories_from_file(args.repos)
    else:
        # Add some default test repositories
        test_repos = [
            {
                'name': 'requests', 
                'url': 'https://github.com/psf/requests.git',
                'language': 'python',
                'priority': 1
            },
            {
                'name': 'axios',
                'url': 'https://github.com/axios/axios.git', 
                'language': 'javascript',
                'priority': 1
            },
            {
                'name': 'fastjson',
                'url': 'https://github.com/alibaba/fastjson.git',
                'language': 'java',
                'priority': 2
            }
        ]
        
        for repo in test_repos:
            daemon.add_repository(**repo)
    
    if args.dry_run:
        print("Configuration validated successfully")
        print(f"Would process {daemon.job_queue.qsize()} repositories with {config['max_workers']} workers")
        return
    
    # Start daemon
    try:
        daemon.start()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        logger.error(f"Daemon error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())