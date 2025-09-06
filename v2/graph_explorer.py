#!/usr/bin/env python3

import os
import sqlite3
import json
import zlib
import gzip
from typing import Dict, List, Any, Optional, Tuple
from flask import Flask, render_template, request, jsonify
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class GraphExplorer:
    def __init__(self, data_path: str = "data/mini_graphs"):
        self.data_path = data_path
        self.repositories = self._discover_repositories()
    
    def _discover_repositories(self) -> List[str]:
        """Discover all available repository databases"""
        repos = []
        if os.path.exists(self.data_path):
            for item in os.listdir(self.data_path):
                repo_path = os.path.join(self.data_path, item)
                if os.path.isdir(repo_path):
                    db_path = os.path.join(repo_path, "graph.db")
                    if os.path.exists(db_path):
                        repos.append(item)
        return sorted(repos)
    
    def _get_db_connection(self, repo_name: str) -> Optional[sqlite3.Connection]:
        """Get database connection for a specific repository"""
        db_path = os.path.join(self.data_path, repo_name, "graph.db")
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            return conn
        return None
    
    def _decompress_json(self, compressed_data: bytes) -> Dict:
        """Decompress and parse JSON data"""
        if not compressed_data:
            return {}
        try:
            # Try gzip first (most likely)
            decompressed = gzip.decompress(compressed_data)
            return json.loads(decompressed.decode('utf-8'))
        except Exception:
            try:
                # Try zlib as fallback
                decompressed = zlib.decompress(compressed_data)
                return json.loads(decompressed.decode('utf-8'))
            except Exception as e:
                logging.error(f"Error decompressing JSON: {e}")
                # Try as plain text
                try:
                    return json.loads(compressed_data.decode('utf-8'))
                except:
                    return {'raw_value': str(compressed_data)}
    
    def get_repository_stats(self, repo_name: str) -> Dict[str, Any]:
        """Get statistics for a specific repository"""
        conn = self._get_db_connection(repo_name)
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor()
            
            # Get basic counts
            cursor.execute("SELECT COUNT(*) FROM nodes")
            node_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM edges")
            edge_count = cursor.fetchone()[0]
            
            # Get node type distribution
            cursor.execute("SELECT node_type, COUNT(*) as count FROM nodes GROUP BY node_type ORDER BY count DESC")
            node_types = dict(cursor.fetchall())
            
            # Get edge type distribution
            cursor.execute("SELECT edge_type, COUNT(*) as count FROM edges GROUP BY edge_type ORDER BY count DESC")
            edge_types = dict(cursor.fetchall())
            
            # Get language distribution
            cursor.execute("SELECT language, COUNT(*) as count FROM nodes WHERE language IS NOT NULL GROUP BY language ORDER BY count DESC")
            languages = dict(cursor.fetchall())
            
            # Get file distribution
            cursor.execute("SELECT file_path, COUNT(*) as count FROM nodes WHERE file_path IS NOT NULL GROUP BY file_path ORDER BY count DESC LIMIT 10")
            top_files = dict(cursor.fetchall())
            
            # Get metadata
            cursor.execute("SELECT key, value FROM graph_metadata")
            metadata_raw = cursor.fetchall()
            metadata = {}
            for key, value in metadata_raw:
                if value:
                    try:
                        metadata[key] = self._decompress_json(value)
                    except:
                        metadata[key] = str(value)
            
            return {
                'repository': repo_name,
                'node_count': node_count,
                'edge_count': edge_count,
                'node_types': node_types,
                'edge_types': edge_types,
                'languages': languages,
                'top_files': top_files,
                'metadata': metadata
            }
        finally:
            conn.close()
    
    def get_nodes(self, repo_name: str, filters: Dict[str, Any] = None, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get nodes with optional filtering"""
        conn = self._get_db_connection(repo_name)
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            
            # Build query with filters
            where_clauses = []
            params = []
            
            if filters:
                if filters.get('node_type'):
                    where_clauses.append("node_type = ?")
                    params.append(filters['node_type'])
                
                if filters.get('language'):
                    where_clauses.append("language = ?")
                    params.append(filters['language'])
                
                if filters.get('file_path'):
                    where_clauses.append("file_path LIKE ?")
                    params.append(f"%{filters['file_path']}%")
                
                if filters.get('name'):
                    where_clauses.append("name LIKE ?")
                    params.append(f"%{filters['name']}%")
                
                if filters.get('code'):
                    where_clauses.append("code LIKE ?")
                    params.append(f"%{filters['code']}%")
            
            where_clause = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
            
            query = f"""
                SELECT id, node_type, name, code, file_path, start_line, end_line, 
                       start_column, end_column, language, properties
                FROM nodes 
                {where_clause}
                ORDER BY file_path, start_line
                LIMIT ? OFFSET ?
            """
            
            params.extend([limit, offset])
            cursor.execute(query, params)
            
            nodes = []
            for row in cursor.fetchall():
                node = dict(row)
                if node['properties']:
                    node['properties'] = self._decompress_json(node['properties'])
                else:
                    node['properties'] = {}
                nodes.append(node)
            
            return nodes
        finally:
            conn.close()
    
    def get_edges(self, repo_name: str, filters: Dict[str, Any] = None, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get edges with optional filtering"""
        conn = self._get_db_connection(repo_name)
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            
            # Build query with filters
            where_clauses = []
            params = []
            
            if filters:
                if filters.get('edge_type'):
                    where_clauses.append("edge_type = ?")
                    params.append(filters['edge_type'])
                
                if filters.get('source_id'):
                    where_clauses.append("source_id = ?")
                    params.append(filters['source_id'])
                
                if filters.get('target_id'):
                    where_clauses.append("target_id = ?")
                    params.append(filters['target_id'])
            
            where_clause = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
            
            query = f"""
                SELECT e.id, e.source_id, e.target_id, e.edge_type, e.properties,
                       s.name as source_name, s.node_type as source_type,
                       t.name as target_name, t.node_type as target_type
                FROM edges e
                LEFT JOIN nodes s ON e.source_id = s.id
                LEFT JOIN nodes t ON e.target_id = t.id
                {where_clause}
                ORDER BY e.edge_type, e.source_id
                LIMIT ? OFFSET ?
            """
            
            params.extend([limit, offset])
            cursor.execute(query, params)
            
            edges = []
            for row in cursor.fetchall():
                edge = dict(row)
                if edge['properties']:
                    edge['properties'] = self._decompress_json(edge['properties'])
                else:
                    edge['properties'] = {}
                edges.append(edge)
            
            return edges
        finally:
            conn.close()
    
    def get_node_relationships(self, repo_name: str, node_id: str) -> Dict[str, List[Dict]]:
        """Get all relationships for a specific node"""
        conn = self._get_db_connection(repo_name)
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor()
            
            # Get outgoing edges
            cursor.execute("""
                SELECT e.id, e.target_id, e.edge_type, e.properties,
                       t.name as target_name, t.node_type as target_type, t.code as target_code
                FROM edges e
                LEFT JOIN nodes t ON e.target_id = t.id
                WHERE e.source_id = ?
                ORDER BY e.edge_type
            """, [node_id])
            
            outgoing = []
            for row in cursor.fetchall():
                edge = dict(row)
                if edge['properties']:
                    edge['properties'] = self._decompress_json(edge['properties'])
                else:
                    edge['properties'] = {}
                outgoing.append(edge)
            
            # Get incoming edges
            cursor.execute("""
                SELECT e.id, e.source_id, e.edge_type, e.properties,
                       s.name as source_name, s.node_type as source_type, s.code as source_code
                FROM edges e
                LEFT JOIN nodes s ON e.source_id = s.id
                WHERE e.target_id = ?
                ORDER BY e.edge_type
            """, [node_id])
            
            incoming = []
            for row in cursor.fetchall():
                edge = dict(row)
                if edge['properties']:
                    edge['properties'] = self._decompress_json(edge['properties'])
                else:
                    edge['properties'] = {}
                incoming.append(edge)
            
            return {
                'outgoing': outgoing,
                'incoming': incoming
            }
        finally:
            conn.close()

# Initialize explorer
explorer = GraphExplorer()

@app.route('/')
def index():
    return render_template('graph_explorer.html', repositories=explorer.repositories)

@app.route('/test')
def test():
    with open('test_api.html', 'r') as f:
        return f.read()

@app.route('/api/repositories')
def api_repositories():
    return jsonify(explorer.repositories)

@app.route('/api/repository/<repo_name>/stats')
def api_repository_stats(repo_name):
    stats = explorer.get_repository_stats(repo_name)
    return jsonify(stats)

@app.route('/api/repository/<repo_name>/nodes')
def api_nodes(repo_name):
    filters = {
        'node_type': request.args.get('node_type'),
        'language': request.args.get('language'),
        'file_path': request.args.get('file_path'),
        'name': request.args.get('name'),
        'code': request.args.get('code')
    }
    # Remove None values
    filters = {k: v for k, v in filters.items() if v}
    
    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))
    
    nodes = explorer.get_nodes(repo_name, filters, limit, offset)
    return jsonify(nodes)

@app.route('/api/repository/<repo_name>/edges')
def api_edges(repo_name):
    filters = {
        'edge_type': request.args.get('edge_type'),
        'source_id': request.args.get('source_id'),
        'target_id': request.args.get('target_id')
    }
    # Remove None values
    filters = {k: v for k, v in filters.items() if v}
    
    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))
    
    edges = explorer.get_edges(repo_name, filters, limit, offset)
    return jsonify(edges)

@app.route('/api/repository/<repo_name>/node/<node_id>/relationships')
def api_node_relationships(repo_name, node_id):
    relationships = explorer.get_node_relationships(repo_name, node_id)
    return jsonify(relationships)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)