#!/usr/bin/env python3
"""
Mini-Graph Store per Repository
===============================

This module implements a lightweight, per-repository graph storage system.
Instead of one massive database, we create individual graph files per repository
for faster access, reduced memory usage, and better scalability.

Key improvements:
1. Individual SQLite files per repository
2. Streaming edge processing to avoid copying large edge collections
3. Memory-efficient graph construction
4. Fast repository-level queries and visualization
"""

import os
import json
import sqlite3
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import hashlib
import pickle
import gzip

from cpg_core import CodePropertyGraph, CPGNode, CPGEdge, NodeType, EdgeType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MiniGraphConfig:
    """Configuration for mini-graph storage."""
    storage_dir: str = "data/mini_graphs"
    compression: bool = True
    batch_size: int = 500
    index_nodes: bool = True
    index_edges: bool = True
    
    @classmethod
    def default(cls) -> 'MiniGraphConfig':
        """Create default configuration."""
        return cls()


class MiniGraphStore:
    """Lightweight per-repository graph storage."""
    
    def __init__(self, repository_name: str, config: MiniGraphConfig = None):
        """Initialize mini-graph store for a specific repository."""
        self.repository_name = repository_name
        self.config = config or MiniGraphConfig.default()
        
        # Create storage directory
        self.storage_path = Path(self.config.storage_dir) / self._safe_repo_name(repository_name)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Database file for this repository
        self.db_path = self.storage_path / "graph.db"
        self.metadata_path = self.storage_path / "metadata.json"
        
        # Initialize database
        self._setup_database()
        
        logger.info(f"Initialized MiniGraphStore for repository: {repository_name}")
    
    def _safe_repo_name(self, name: str) -> str:
        """Convert repository name to filesystem-safe string."""
        # Replace problematic characters
        safe_name = name.replace("/", "_").replace("\\", "_").replace(":", "_")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-.")
        return safe_name[:100]  # Limit length
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        try:
            # Optimize for performance
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA cache_size = -32000")  # 32MB cache
            conn.execute("PRAGMA temp_store = MEMORY")
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _setup_database(self):
        """Setup repository-specific database schema."""
        schema_queries = [
            """CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                name TEXT,
                code TEXT,
                file_path TEXT,
                start_line INTEGER,
                end_line INTEGER,
                start_column INTEGER,
                end_column INTEGER,
                language TEXT,
                properties BLOB,  -- Compressed JSON
                created_at REAL DEFAULT (julianday('now'))
            )""",
            
            """CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                properties BLOB,  -- Compressed JSON
                created_at REAL DEFAULT (julianday('now')),
                FOREIGN KEY (source_id) REFERENCES nodes (id),
                FOREIGN KEY (target_id) REFERENCES nodes (id)
            )""",
            
            """CREATE TABLE IF NOT EXISTS graph_metadata (
                key TEXT PRIMARY KEY,
                value BLOB,
                updated_at REAL DEFAULT (julianday('now'))
            )"""
        ]
        
        with self.get_connection() as conn:
            for query in schema_queries:
                conn.execute(query)
            
            # Create indexes if enabled
            if self.config.index_nodes:
                index_queries = [
                    "CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes (node_type)",
                    "CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes (name)",
                    "CREATE INDEX IF NOT EXISTS idx_nodes_file ON nodes (file_path)",
                    "CREATE INDEX IF NOT EXISTS idx_nodes_language ON nodes (language)",
                    "CREATE INDEX IF NOT EXISTS idx_nodes_line ON nodes (start_line)"
                ]
                for query in index_queries:
                    conn.execute(query)
            
            if self.config.index_edges:
                edge_index_queries = [
                    "CREATE INDEX IF NOT EXISTS idx_edges_type ON edges (edge_type)",
                    "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges (source_id)",
                    "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges (target_id)"
                ]
                for query in edge_index_queries:
                    conn.execute(query)
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data for storage."""
        if not self.config.compression:
            return json.dumps(data).encode('utf-8')
        
        json_bytes = json.dumps(data).encode('utf-8')
        return gzip.compress(json_bytes)
    
    def _decompress_data(self, data: bytes) -> Any:
        """Decompress data from storage."""
        if not self.config.compression:
            return json.loads(data.decode('utf-8'))
        
        try:
            decompressed = gzip.decompress(data)
            return json.loads(decompressed.decode('utf-8'))
        except Exception:
            # Fallback for uncompressed data
            return json.loads(data.decode('utf-8'))
    
    def store_cpg_streaming(self, cpg: CodePropertyGraph, commit_hash: str = "") -> bool:
        """
        Store CPG using streaming approach to avoid memory issues.
        
        Args:
            cpg: Code Property Graph to store
            commit_hash: Git commit hash
            
        Returns:
            True if successful, False otherwise
        """
        try:
            start_time = time.time()
            
            # Store nodes first
            nodes_stored = self._store_nodes_streaming(cpg)
            
            # Store edges using streaming approach
            edges_stored = self._store_edges_streaming(cpg)
            
            # Store metadata
            self._store_graph_metadata(cpg, commit_hash, nodes_stored, edges_stored)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Stored CPG for {self.repository_name}: "
                       f"{nodes_stored} nodes, {edges_stored} edges in {processing_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store CPG for {self.repository_name}: {e}")
            return False
    
    def _store_nodes_streaming(self, cpg: CodePropertyGraph) -> int:
        """Store nodes in batches using streaming approach."""
        nodes_stored = 0
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Process nodes in batches
            node_batch = []
            for node in cpg.nodes.values():
                # Prepare node data
                properties_blob = self._compress_data(node.properties) if node.properties else b''
                
                node_data = (
                    node.id,
                    node.node_type.value,
                    node.name,
                    (node.code or '')[:2000],  # Limit code length
                    node.file_path,
                    node.start_line,
                    node.end_line,
                    node.start_column,
                    node.end_column,
                    node.language,
                    properties_blob
                )
                
                node_batch.append(node_data)
                
                # Insert batch when it reaches configured size
                if len(node_batch) >= self.config.batch_size:
                    self._insert_node_batch(cursor, node_batch)
                    nodes_stored += len(node_batch)
                    node_batch = []
            
            # Insert remaining nodes
            if node_batch:
                self._insert_node_batch(cursor, node_batch)
                nodes_stored += len(node_batch)
        
        return nodes_stored
    
    def _insert_node_batch(self, cursor, node_batch):
        """Insert a batch of nodes."""
        query = """
        INSERT OR REPLACE INTO nodes 
        (id, node_type, name, code, file_path, start_line, end_line, 
         start_column, end_column, language, properties)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.executemany(query, node_batch)
    
    def _store_edges_streaming(self, cpg: CodePropertyGraph) -> int:
        """Store edges using streaming approach to avoid copying."""
        edges_stored = 0
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Process edges in batches without copying the entire collection
            edge_batch = []
            edge_iter = iter(cpg.edges.values())
            
            try:
                while True:
                    # Get next edge without copying
                    try:
                        edge = next(edge_iter)
                    except StopIteration:
                        break
                    
                    # Prepare edge data
                    properties_blob = self._compress_data(edge.properties) if edge.properties else b''
                    
                    edge_data = (
                        edge.id,
                        edge.source_id,
                        edge.target_id,
                        edge.edge_type.value,
                        properties_blob
                    )
                    
                    edge_batch.append(edge_data)
                    
                    # Insert batch when it reaches configured size
                    if len(edge_batch) >= self.config.batch_size:
                        self._insert_edge_batch(cursor, edge_batch)
                        edges_stored += len(edge_batch)
                        edge_batch = []
                        
                        # Optional: yield control to avoid blocking
                        if edges_stored % (self.config.batch_size * 10) == 0:
                            conn.commit()  # Intermediate commit
                
                # Insert remaining edges
                if edge_batch:
                    self._insert_edge_batch(cursor, edge_batch)
                    edges_stored += len(edge_batch)
                    
            except Exception as e:
                logger.error(f"Error during edge streaming: {e}")
                raise
        
        return edges_stored
    
    def _insert_edge_batch(self, cursor, edge_batch):
        """Insert a batch of edges."""
        query = """
        INSERT OR REPLACE INTO edges 
        (id, source_id, target_id, edge_type, properties)
        VALUES (?, ?, ?, ?, ?)
        """
        cursor.executemany(query, edge_batch)
    
    def _store_graph_metadata(self, cpg: CodePropertyGraph, commit_hash: str, 
                             nodes_stored: int, edges_stored: int):
        """Store graph metadata."""
        metadata = {
            'repository_name': self.repository_name,
            'commit_hash': commit_hash,
            'total_nodes': nodes_stored,
            'total_edges': edges_stored,
            'languages': list(set(node.language for node in cpg.nodes.values() if node.language)),
            'file_count': len(set(node.file_path for node in cpg.nodes.values() if node.file_path)),
            'created_at': time.time(),
            'cpg_metadata': cpg.metadata
        }
        
        # Store in database
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for key, value in metadata.items():
                compressed_value = self._compress_data(value)
                cursor.execute(
                    "INSERT OR REPLACE INTO graph_metadata (key, value) VALUES (?, ?)",
                    (key, compressed_value)
                )
        
        # Also save as JSON file for quick access
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save metadata JSON: {e}")
    
    def query_nodes(self, node_type: str = None, name_pattern: str = None, 
                   file_path: str = None, limit: int = 100) -> List[Dict]:
        """Query nodes with optional filters."""
        query = "SELECT id, node_type, name, file_path, start_line, language FROM nodes WHERE 1=1"
        params = []
        
        if node_type:
            query += " AND node_type = ?"
            params.append(node_type)
        
        if name_pattern:
            query += " AND name LIKE ?"
            params.append(f"%{name_pattern}%")
        
        if file_path:
            query += " AND file_path = ?"
            params.append(file_path)
        
        query += f" LIMIT {limit}"
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def query_edges(self, edge_type: str = None, source_id: str = None, 
                   target_id: str = None, limit: int = 100) -> List[Dict]:
        """Query edges with optional filters."""
        query = "SELECT id, source_id, target_id, edge_type FROM edges WHERE 1=1"
        params = []
        
        if edge_type:
            query += " AND edge_type = ?"
            params.append(edge_type)
        
        if source_id:
            query += " AND source_id = ?"
            params.append(source_id)
        
        if target_id:
            query += " AND target_id = ?"
            params.append(target_id)
        
        query += f" LIMIT {limit}"
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_repository_stats(self) -> Dict[str, Any]:
        """Get statistics for this repository."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Node statistics
            cursor.execute("SELECT node_type, COUNT(*) FROM nodes GROUP BY node_type")
            node_stats = dict(cursor.fetchall())
            
            # Edge statistics  
            cursor.execute("SELECT edge_type, COUNT(*) FROM edges GROUP BY edge_type")
            edge_stats = dict(cursor.fetchall())
            
            # Total counts
            cursor.execute("SELECT COUNT(*) FROM nodes")
            total_nodes = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM edges") 
            total_edges = cursor.fetchone()[0]
            
            # File statistics
            cursor.execute("SELECT language, COUNT(DISTINCT file_path) FROM nodes GROUP BY language")
            language_stats = dict(cursor.fetchall())
            
        return {
            'repository_name': self.repository_name,
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'node_types': node_stats,
            'edge_types': edge_stats,
            'languages': language_stats,
            'storage_path': str(self.storage_path),
            'db_size_mb': self.get_database_size_mb()
        }
    
    def get_database_size_mb(self) -> float:
        """Get database size in MB."""
        try:
            size_bytes = self.db_path.stat().st_size
            return size_bytes / (1024 * 1024)
        except:
            return 0.0
    
    def export_for_visualization(self, output_file: str = None) -> str:
        """Export graph data in format suitable for visualization."""
        if output_file is None:
            output_file = str(self.storage_path / "visualization.json")
        
        # Query all nodes and edges
        nodes = self.query_nodes(limit=5000)  # Reasonable limit for visualization
        edges = self.query_edges(limit=10000)
        
        # Create visualization format
        viz_data = {
            'repository': self.repository_name,
            'stats': self.get_repository_stats(),
            'nodes': nodes,
            'edges': edges
        }
        
        with open(output_file, 'w') as f:
            json.dump(viz_data, f, indent=2, default=str)
        
        logger.info(f"Exported visualization data to: {output_file}")
        return output_file
    
    def clear_repository(self):
        """Clear all data for this repository."""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM edges")
            conn.execute("DELETE FROM nodes") 
            conn.execute("DELETE FROM graph_metadata")
        
        logger.info(f"Cleared repository data: {self.repository_name}")
    
    def close(self):
        """Close the mini-graph store and perform cleanup."""
        # Since we use context managers for connections, no explicit cleanup needed
        # But log the closure for monitoring
        logger.debug(f"Closed mini-graph store for repository: {self.repository_name}")


class MiniGraphManager:
    """Manager for multiple mini-graph stores."""
    
    def __init__(self, config: MiniGraphConfig = None):
        """Initialize mini-graph manager."""
        self.config = config or MiniGraphConfig.default()
        self.storage_dir = Path(self.config.storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized MiniGraphManager with storage: {self.storage_dir}")
    
    def get_store(self, repository_name: str) -> MiniGraphStore:
        """Get mini-graph store for a repository."""
        return MiniGraphStore(repository_name, self.config)
    
    def list_repositories(self) -> List[str]:
        """List all repositories with mini-graphs."""
        repositories = []
        for repo_dir in self.storage_dir.iterdir():
            if repo_dir.is_dir() and (repo_dir / "graph.db").exists():
                repositories.append(repo_dir.name)
        return sorted(repositories)
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics across all repositories."""
        repositories = self.list_repositories()
        total_stats = {
            'total_repositories': len(repositories),
            'total_nodes': 0,
            'total_edges': 0,
            'total_size_mb': 0.0,
            'languages': set(),
            'repositories': []
        }
        
        for repo_name in repositories:
            try:
                store = self.get_store(repo_name)
                repo_stats = store.get_repository_stats()
                
                total_stats['total_nodes'] += repo_stats['total_nodes']
                total_stats['total_edges'] += repo_stats['total_edges'] 
                total_stats['total_size_mb'] += repo_stats['db_size_mb']
                total_stats['languages'].update(repo_stats['languages'].keys())
                total_stats['repositories'].append({
                    'name': repo_name,
                    'nodes': repo_stats['total_nodes'],
                    'edges': repo_stats['total_edges'],
                    'size_mb': repo_stats['db_size_mb']
                })
                
            except Exception as e:
                logger.error(f"Error getting stats for {repo_name}: {e}")
        
        total_stats['languages'] = list(total_stats['languages'])
        return total_stats


def create_mini_graph_store(repository_name: str) -> MiniGraphStore:
    """Create mini-graph store with default configuration."""
    return MiniGraphStore(repository_name, MiniGraphConfig.default())


def create_mini_graph_manager() -> MiniGraphManager:
    """Create mini-graph manager with default configuration."""
    return MiniGraphManager(MiniGraphConfig.default())