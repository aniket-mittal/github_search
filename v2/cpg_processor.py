#!/usr/bin/env python3
"""
CPG Batch Processor

This script provides batch processing capabilities for generating Code Property Graphs
across entire codebases. It includes robust error handling, progress tracking, and
memory management to handle large repositories.
"""

import os
import json
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing
import gc

from cpg_core import CPGBuilder, get_supported_languages, get_file_extensions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for batch processing."""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    processing_time: float = 0.0
    languages_processed: Dict[str, int] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.languages_processed is None:
            self.languages_processed = {}
        if self.errors is None:
            self.errors = []


class CPGBatchProcessor:
    """Batch processor for generating CPGs across large codebases."""
    
    def __init__(self, max_workers: int = None, chunk_size: int = 100, 
                 max_file_size: int = 10 * 1024 * 1024):  # 10MB default
        """
        Initialize the batch processor.
        
        Args:
            max_workers: Maximum number of worker processes
            chunk_size: Number of files to process in each batch
            max_file_size: Maximum file size to process (in bytes)
        """
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)
        self.chunk_size = chunk_size
        self.max_file_size = max_file_size
        self.cpg_builder = CPGBuilder()
        self.supported_extensions = set(get_file_extensions().keys())
        
        logger.info(f"Initialized CPG Batch Processor with {self.max_workers} workers")
    
    def discover_files(self, root_path: str, include_patterns: List[str] = None,
                      exclude_patterns: List[str] = None) -> List[str]:
        """
        Discover all relevant source files in the given directory.
        
        Args:
            root_path: Root directory to search
            include_patterns: File patterns to include (e.g., ['*.py', '*.js'])
            exclude_patterns: File patterns to exclude (e.g., ['*test*', '*__pycache__*'])
        
        Returns:
            List of file paths to process
        """
        root_path = Path(root_path)
        discovered_files = []
        
        # Default exclude patterns
        default_excludes = [
            '*/__pycache__/*',
            '*/node_modules/*',
            '*/venv/*',
            '*/env/*',
            '*/.git/*',
            '*/.svn/*',
            '*/build/*',
            '*/dist/*',
            '*/target/*',
            '*/.idea/*',
            '*/.vscode/*',
            '*/bin/*',
            '*/obj/*',
            '*/Debug/*',
            '*/Release/*',
            '*/test*',
            '*/Test*',
            '*test*',
            '*Test*',
            '*.min.js',
            '*.min.css',
            '*.bundle.js',
            '*.bundle.css',
        ]
        
        exclude_patterns = (exclude_patterns or []) + default_excludes
        
        logger.info(f"Discovering files in {root_path}")
        
        # If include patterns specified, use them; otherwise use supported extensions
        if include_patterns:
            patterns_to_search = include_patterns
        else:
            patterns_to_search = [f"*{ext}" for ext in self.supported_extensions]
        
        # Discover files
        for pattern in patterns_to_search:
            for file_path in root_path.rglob(pattern):
                if file_path.is_file():
                    # Check exclude patterns
                    should_exclude = False
                    for exclude_pattern in exclude_patterns:
                        if file_path.match(exclude_pattern):
                            should_exclude = True
                            break
                    
                    if not should_exclude:
                        # Check file size
                        try:
                            if file_path.stat().st_size <= self.max_file_size:
                                discovered_files.append(str(file_path))
                            else:
                                logger.warning(f"Skipping large file: {file_path} "
                                             f"({file_path.stat().st_size} bytes)")
                        except OSError:
                            logger.warning(f"Could not stat file: {file_path}")
        
        discovered_files = sorted(list(set(discovered_files)))  # Remove duplicates and sort
        logger.info(f"Discovered {len(discovered_files)} files to process")
        
        return discovered_files
    
    def process_single_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single file and return results.
        
        Args:
            file_path: Path to the file to process
        
        Returns:
            Dictionary containing processing results
        """
        try:
            start_time = time.time()
            
            # Build CPG
            cpg = self.cpg_builder.build_cpg(file_path)
            cpg_error = cpg.metadata.get('error') if isinstance(cpg.metadata, dict) else None
            
            processing_time = time.time() - start_time
            
            return {
                'file_path': file_path,
                'success': False if cpg_error else True,
                'nodes': len(cpg.nodes),
                'edges': len(cpg.edges),
                'language': cpg.metadata.get('language', 'unknown'),
                'processing_time': processing_time,
                'error': str(cpg_error) if cpg_error else None,
                'cpg_data': {
                    'nodes': {node_id: {
                        'id': node.id,
                        'node_type': node.node_type.value,
                        'name': node.name,
                        'code': node.code[:200] if len(node.code) > 200 else node.code,  # Truncate for memory
                        'file_path': node.file_path,
                        'start_line': node.start_line,
                        'end_line': node.end_line,
                        'language': node.language,
                        'properties': node.properties
                    } for node_id, node in cpg.nodes.items()},
                    'edges': {edge_id: {
                        'id': edge.id,
                        'source_id': edge.source_id,
                        'target_id': edge.target_id,
                        'edge_type': edge.edge_type.value,
                        'properties': edge.properties
                    } for edge_id, edge in cpg.edges.items()},
                    'metadata': cpg.metadata
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return {
                'file_path': file_path,
                'success': False,
                'nodes': 0,
                'edges': 0,
                'language': 'unknown',
                'processing_time': 0.0,
                'error': str(e),
                'cpg_data': None
            }
    
    def process_batch(self, file_paths: List[str], output_dir: str = None,
                     save_individual: bool = True, save_combined: bool = True,
                     progress_callback: callable = None) -> ProcessingStats:
        """
        Process a batch of files.
        
        Args:
            file_paths: List of file paths to process
            output_dir: Directory to save CPG files
            save_individual: Whether to save individual CPG files
            save_combined: Whether to save a combined summary
            progress_callback: Optional callback function for progress updates
        
        Returns:
            ProcessingStats object with processing statistics
        """
        stats = ProcessingStats()
        stats.total_files = len(file_paths)
        
        start_time = time.time()
        
        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Process files in chunks to manage memory
        all_results = []
        
        for i in range(0, len(file_paths), self.chunk_size):
            chunk = file_paths[i:i + self.chunk_size]
            chunk_results = self._process_chunk(chunk, progress_callback, stats)
            all_results.extend(chunk_results)
            
            # Save individual files if requested
            if output_dir and save_individual:
                self._save_chunk_results(chunk_results, output_path)
            
            # Force garbage collection to manage memory
            gc.collect()
        
        # Calculate final statistics
        stats.processing_time = time.time() - start_time
        
        # Save combined results if requested
        if output_dir and save_combined:
            self._save_combined_results(all_results, output_path, stats)
        
        # Log final statistics
        self._log_final_stats(stats)
        
        return stats
    
    def _process_chunk(self, file_paths: List[str], progress_callback: callable,
                      stats: ProcessingStats) -> List[Dict[str, Any]]:
        """Process a chunk of files using multiprocessing."""
        chunk_results = []
        
        if self.max_workers == 1:
            # Single-threaded processing
            for file_path in file_paths:
                result = self.process_single_file(file_path)
                chunk_results.append(result)
                self._update_stats(result, stats)
                
                if progress_callback:
                    progress_callback(stats.processed_files + stats.failed_files, 
                                    stats.total_files)
        else:
            # Multi-threaded processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self.process_single_file, file_path): file_path
                    for file_path in file_paths
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_file):
                    result = future.result()
                    chunk_results.append(result)
                    self._update_stats(result, stats)
                    
                    if progress_callback:
                        progress_callback(stats.processed_files + stats.failed_files, 
                                        stats.total_files)
        
        return chunk_results
    
    def _update_stats(self, result: Dict[str, Any], stats: ProcessingStats):
        """Update processing statistics with a single result."""
        if result['success']:
            stats.processed_files += 1
            stats.total_nodes += result['nodes']
            stats.total_edges += result['edges']
            
            language = result['language']
            stats.languages_processed[language] = stats.languages_processed.get(language, 0) + 1
        else:
            stats.failed_files += 1
            stats.errors.append(f"{result['file_path']}: {result['error']}")
    
    def _save_chunk_results(self, results: List[Dict[str, Any]], output_path: Path):
        """Save individual CPG files for a chunk of results."""
        def _sanitize_for_json(obj):
            """Recursively convert non-JSON-serializable types (e.g., set, tuple) to serializable ones."""
            if isinstance(obj, dict):
                return {k: _sanitize_for_json(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize_for_json(v) for v in obj]
            if isinstance(obj, tuple):
                return [_sanitize_for_json(v) for v in obj]
            if isinstance(obj, set):
                # Convert sets to sorted lists for determinism
                try:
                    return sorted(_sanitize_for_json(v) for v in obj)
                except Exception:
                    return list(_sanitize_for_json(v) for v in obj)
            return obj

        for result in results:
            if result['success'] and result['cpg_data']:
                file_name = Path(result['file_path']).name + '.cpg.json'
                output_file = output_path / file_name
                
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(_sanitize_for_json(result['cpg_data']), f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"Failed to save CPG for {result['file_path']}: {e}")
    
    def _save_combined_results(self, all_results: List[Dict[str, Any]], 
                              output_path: Path, stats: ProcessingStats):
        """Save combined processing results and statistics."""
        # Save summary statistics
        summary_file = output_path / 'processing_summary.json'
        summary_data = {
            'total_files': stats.total_files,
            'processed_files': stats.processed_files,
            'failed_files': stats.failed_files,
            'total_nodes': stats.total_nodes,
            'total_edges': stats.total_edges,
            'processing_time': stats.processing_time,
            'languages_processed': stats.languages_processed,
            'errors': stats.errors[:100],  # Limit errors to first 100
            'files_per_second': stats.processed_files / stats.processing_time if stats.processing_time > 0 else 0,
            'nodes_per_file': stats.total_nodes / stats.processed_files if stats.processed_files > 0 else 0,
            'edges_per_file': stats.total_edges / stats.processed_files if stats.processed_files > 0 else 0,
        }
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved processing summary to {summary_file}")
        except Exception as e:
            logger.error(f"Failed to save processing summary: {e}")
        
        # Save file-level results (without full CPG data to save space)
        results_file = output_path / 'file_results.json'
        file_results = []
        
        for result in all_results:
            file_result = {
                'file_path': result['file_path'],
                'success': result['success'],
                'nodes': result['nodes'],
                'edges': result['edges'],
                'language': result['language'],
                'processing_time': result['processing_time'],
                'error': result['error']
            }
            file_results.append(file_result)
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(file_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved file results to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save file results: {e}")
    
    def _log_final_stats(self, stats: ProcessingStats):
        """Log final processing statistics."""
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total files: {stats.total_files}")
        logger.info(f"Successfully processed: {stats.processed_files}")
        logger.info(f"Failed: {stats.failed_files}")
        logger.info(f"Total nodes generated: {stats.total_nodes:,}")
        logger.info(f"Total edges generated: {stats.total_edges:,}")
        logger.info(f"Processing time: {stats.processing_time:.2f} seconds")
        
        if stats.processing_time > 0:
            logger.info(f"Files per second: {stats.processed_files / stats.processing_time:.2f}")
        
        if stats.processed_files > 0:
            logger.info(f"Average nodes per file: {stats.total_nodes / stats.processed_files:.1f}")
            logger.info(f"Average edges per file: {stats.total_edges / stats.processed_files:.1f}")
        
        logger.info("\nLanguages processed:")
        for language, count in sorted(stats.languages_processed.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {language}: {count} files")
        
        if stats.errors:
            logger.warning(f"\nFirst 10 errors:")
            for error in stats.errors[:10]:
                logger.warning(f"  {error}")


def progress_callback(current: int, total: int):
    """Simple progress callback function."""
    percentage = (current / total) * 100 if total > 0 else 0
    print(f"\rProgress: {current}/{total} ({percentage:.1f}%)", end='', flush=True)


def main():
    """Main entry point for the CPG batch processor."""
    parser = argparse.ArgumentParser(description='Batch process source code to generate CPGs')
    parser.add_argument('input_path', help='Input directory or file to process')
    parser.add_argument('-o', '--output', help='Output directory for CPG files')
    parser.add_argument('-w', '--workers', type=int, help='Number of worker processes')
    parser.add_argument('-c', '--chunk-size', type=int, default=100, 
                       help='Number of files to process in each batch')
    parser.add_argument('--max-file-size', type=int, default=10*1024*1024,
                       help='Maximum file size to process (bytes)')
    parser.add_argument('--include', nargs='+', help='File patterns to include')
    parser.add_argument('--exclude', nargs='+', help='File patterns to exclude')
    parser.add_argument('--no-individual', action='store_true', 
                       help='Do not save individual CPG files')
    parser.add_argument('--no-combined', action='store_true',
                       help='Do not save combined results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize processor
    processor = CPGBatchProcessor(
        max_workers=args.workers,
        chunk_size=args.chunk_size,
        max_file_size=args.max_file_size
    )
    
    # Discover files
    if os.path.isfile(args.input_path):
        file_paths = [args.input_path]
    else:
        file_paths = processor.discover_files(
            args.input_path,
            include_patterns=args.include,
            exclude_patterns=args.exclude
        )
    
    if not file_paths:
        logger.error("No files found to process")
        return 1
    
    # Process files
    try:
        stats = processor.process_batch(
            file_paths,
            output_dir=args.output,
            save_individual=not args.no_individual,
            save_combined=not args.no_combined,
            progress_callback=progress_callback if not args.verbose else None
        )
        
        print()  # New line after progress
        
        # Return appropriate exit code
        return 0 if stats.failed_files == 0 else 1
        
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
