#!/usr/bin/env python3
"""
Code Property Graph (CPG) Core System

This module provides the core infrastructure for building Code Property Graphs
that include Abstract Syntax Trees (AST), Control Flow Graphs (CFG), and 
Data Flow Graphs (DFG) for multiple programming languages.

The system is designed to be highly accurate and extensible, supporting
all major programming languages found in GitHub repositories.
"""

import os
import ast
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import re
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the CPG."""
    # AST Node Types
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    PARAMETER = "parameter"
    LITERAL = "literal"
    OPERATOR = "operator"
    CALL = "call"
    IMPORT = "import"
    ASSIGNMENT = "assignment"
    CONDITION = "condition"
    LOOP = "loop"
    RETURN = "return"
    EXCEPTION = "exception"
    
    # CFG Node Types
    ENTRY = "entry"
    EXIT = "exit"
    BASIC_BLOCK = "basic_block"
    BRANCH = "branch"
    MERGE = "merge"
    
    # DFG Node Types
    DATA_SOURCE = "data_source"
    DATA_SINK = "data_sink"
    DATA_FLOW = "data_flow"


class EdgeType(Enum):
    """Types of edges in the CPG."""
    # AST Edge Types
    AST_CHILD = "ast_child"
    AST_PARENT = "ast_parent"
    
    # CFG Edge Types
    CONTROL_FLOW = "control_flow"
    CONDITIONAL_TRUE = "conditional_true"
    CONDITIONAL_FALSE = "conditional_false"
    EXCEPTION_FLOW = "exception_flow"
    
    # DFG Edge Types
    DATA_DEPENDENCY = "data_dependency"
    DEFINITION_USE = "def_use"
    USE_DEFINITION = "use_def"
    
    # Semantic Edge Types
    CALL_EDGE = "call"
    INHERITANCE = "inheritance"
    IMPLEMENTS = "implements"
    TYPE_RELATION = "type_relation"


@dataclass
class CPGNode:
    """Represents a node in the Code Property Graph."""
    id: str
    node_type: NodeType
    name: str
    code: str = ""
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    start_column: int = 0
    end_column: int = 0
    language: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.id:
            # Generate ID based on content and position
            content = f"{self.file_path}:{self.start_line}:{self.start_column}:{self.name}:{self.code}"
            self.id = hashlib.md5(content.encode()).hexdigest()[:16]


@dataclass
class CPGEdge:
    """Represents an edge in the Code Property Graph."""
    id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.id:
            # Generate ID based on source, target, and type
            content = f"{self.source_id}:{self.target_id}:{self.edge_type.value}"
            self.id = hashlib.md5(content.encode()).hexdigest()[:16]


@dataclass
class CodePropertyGraph:
    """Complete Code Property Graph structure."""
    nodes: Dict[str, CPGNode] = field(default_factory=dict)
    edges: Dict[str, CPGEdge] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, node: CPGNode) -> str:
        """Add a node to the CPG."""
        self.nodes[node.id] = node
        return node.id
    
    def add_edge(self, edge: CPGEdge) -> str:
        """Add an edge to the CPG."""
        # Avoid inserting duplicate edges with same source, target, and type
        existing_edges = list(self.edges.values())  # Create snapshot first
        for existing_edge in existing_edges:
            if (existing_edge.source_id == edge.source_id and
                existing_edge.target_id == edge.target_id and
                existing_edge.edge_type == edge.edge_type):
                return existing_edge.id

        self.edges[edge.id] = edge

        # Maintain AST parent symmetry automatically
        if edge.edge_type == EdgeType.AST_CHILD:
            # Create reverse AST_PARENT edge if it does not already exist
            reverse_exists = False
            # Use the same snapshot to avoid iteration issues
            for existing_edge in existing_edges:
                if (existing_edge.source_id == edge.target_id and
                    existing_edge.target_id == edge.source_id and
                    existing_edge.edge_type == EdgeType.AST_PARENT):
                    reverse_exists = True
                    break
            if not reverse_exists:
                reverse_edge = CPGEdge(
                    id="",
                    source_id=edge.target_id,
                    target_id=edge.source_id,
                    edge_type=EdgeType.AST_PARENT,
                    properties={}
                )
                self.edges[reverse_edge.id] = reverse_edge

        return edge.id
    
    def get_node(self, node_id: str) -> Optional[CPGNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[CPGEdge]:
        """Get an edge by ID."""
        return self.edges.get(edge_id)
    
    def get_outgoing_edges(self, node_id: str, edge_type: EdgeType = None) -> List[CPGEdge]:
        """Get all outgoing edges from a node."""
        edges = []
        # Snapshot values to avoid 'dictionary changed size during iteration'
        for edge in list(self.edges.values()):
            if edge.source_id == node_id:
                if edge_type is None or edge.edge_type == edge_type:
                    edges.append(edge)
        return edges
    
    def get_incoming_edges(self, node_id: str, edge_type: EdgeType = None) -> List[CPGEdge]:
        """Get all incoming edges to a node."""
        edges = []
        # Snapshot values to avoid 'dictionary changed size during iteration'
        for edge in list(self.edges.values()):
            if edge.target_id == node_id:
                if edge_type is None or edge.edge_type == edge_type:
                    edges.append(edge)
        return edges
    
    def get_children(self, node_id: str) -> List[CPGNode]:
        """Get all child nodes in AST."""
        children = []
        for edge in self.get_outgoing_edges(node_id, EdgeType.AST_CHILD):
            child = self.get_node(edge.target_id)
            if child:
                children.append(child)
        return children
    
    def get_parent(self, node_id: str) -> Optional[CPGNode]:
        """Get parent node in AST."""
        edges = self.get_incoming_edges(node_id, EdgeType.AST_PARENT)
        if edges:
            return self.get_node(edges[0].source_id)
        return None


class LanguageDetector:
    """Advanced language detection based on file extensions and content analysis."""
    
    LANGUAGE_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'react',
        '.tsx': 'react-ts',
        '.java': 'java',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.php': 'php',
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.kts': 'kotlin',
        '.scala': 'scala',
        '.clj': 'clojure',
        '.cljs': 'clojure',
        '.hs': 'haskell',
        '.ml': 'ocaml',
        '.mli': 'ocaml',
        '.fs': 'fsharp',
        '.fsi': 'fsharp',
        '.r': 'r',
        '.R': 'r',
        '.m': 'matlab',
        '.sh': 'bash',
        '.bash': 'bash',
        '.zsh': 'zsh',
        '.sql': 'sql',
        '.html': 'html',
        '.htm': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.sass': 'sass',
        '.less': 'less',
        '.xml': 'xml',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.toml': 'toml',
        '.ini': 'ini',
        '.cfg': 'config',
        '.conf': 'config',
        '.md': 'markdown',
        '.txt': 'text',
        '.lua': 'lua',
        '.pl': 'perl',
        '.pm': 'perl',
        '.vim': 'vim',
        '.vhd': 'vhdl',
        '.vhdl': 'vhdl',
        '.v': 'verilog',
        '.sv': 'systemverilog',
        '.dart': 'dart',
        '.elm': 'elm',
        '.ex': 'elixir',
        '.exs': 'elixir',
        '.erl': 'erlang',
        '.hrl': 'erlang',
        '.f': 'fortran',
        '.f90': 'fortran',
        '.f95': 'fortran',
        '.groovy': 'groovy',
        '.gradle': 'gradle',
        '.hx': 'haxe',
        '.jl': 'julia',
        '.lisp': 'lisp',
        '.lsp': 'lisp',
        '.nim': 'nim',
        '.nims': 'nim',
        '.pas': 'pascal',
        '.pp': 'pascal',
        '.proto': 'protobuf',
        '.rkt': 'racket',
        '.scm': 'scheme',
        '.st': 'smalltalk',
        '.tcl': 'tcl',
        '.vb': 'vb',
        '.vbs': 'vbscript',
        '.zig': 'zig',
    }
    
    # Content-based detection patterns
    CONTENT_PATTERNS = {
        'python': [
            r'def\s+\w+\s*\(',
            r'class\s+\w+\s*:',
            r'import\s+\w+',
            r'from\s+\w+\s+import',
            r'if\s+__name__\s*==\s*[\'"]__main__[\'"]',
        ],
        'javascript': [
            r'function\s+\w+\s*\(',
            r'var\s+\w+\s*=',
            r'let\s+\w+\s*=',
            r'const\s+\w+\s*=',
            r'=>',
            r'require\s*\(',
        ],
        'typescript': [
            r'interface\s+\w+',
            r'type\s+\w+\s*=',
            r':\s*\w+\s*=',
            r'export\s+\w+',
            r'import\s+.*from',
        ],
        'java': [
            r'public\s+class\s+\w+',
            r'private\s+\w+',
            r'protected\s+\w+',
            r'import\s+java\.',
            r'package\s+\w+',
        ],
        'cpp': [
            r'#include\s*<.*>',
            r'namespace\s+\w+',
            r'class\s+\w+',
            r'std::',
            r'template\s*<',
        ],
        'c': [
            r'#include\s*<.*\.h>',
            r'int\s+main\s*\(',
            r'struct\s+\w+',
            r'typedef\s+',
        ],
        'csharp': [
            r'using\s+System',
            r'namespace\s+\w+',
            r'public\s+class\s+\w+',
            r'private\s+\w+',
            r'protected\s+\w+',
        ],
        'php': [
            r'<\?php',
            r'function\s+\w+\s*\(',
            r'\$\w+\s*=',
            r'class\s+\w+',
            r'require_once',
        ],
        'ruby': [
            r'def\s+\w+',
            r'class\s+\w+',
            r'module\s+\w+',
            r'require\s+[\'"]',
            r'end\s*$',
        ],
        'go': [
            r'package\s+\w+',
            r'func\s+\w+\s*\(',
            r'import\s*\(',
            r'type\s+\w+\s+struct',
            r'var\s+\w+\s+\w+',
        ],
        'rust': [
            r'fn\s+\w+\s*\(',
            r'struct\s+\w+',
            r'impl\s+\w+',
            r'use\s+\w+',
            r'let\s+\w+\s*=',
        ],
        'objective-c': [
            r'@interface\s+\w+',
            r'@implementation\s+\w+',
            r'\[[^\]]+\s+\w+(?::[^\]]+)?\]',
            r'#import\s*[<"][^>"]+[>"]',
        ],
    }
    
    @classmethod
    def detect_language(cls, file_path: str, content: str = None) -> str:
        """Detect programming language from file path and content."""
        # First try extension-based detection
        ext = Path(file_path).suffix.lower()
        if ext in cls.LANGUAGE_EXTENSIONS:
            detected = cls.LANGUAGE_EXTENSIONS[ext]
            
            # For ambiguous extensions, use content analysis
            if content and ext in ['.h', '.m', '.mm']:
                if cls._matches_patterns(content, 'cpp'):
                    return 'cpp'
                elif cls._matches_patterns(content, 'c'):
                    return 'c'
                # Objective-C heuristics
                elif ('objective-c' in content.lower() or
                      '@interface' in content or
                      '@implementation' in content or
                      re.search(r'\[[^\]]+\s+\w+(?::[^\]]+)?\]', content)):
                    return 'objective-c'
            
            return detected
        
        # Fallback to content-based detection
        if content:
            return cls._detect_from_content(content)
        
        return 'unknown'
    
    @classmethod
    def _detect_from_content(cls, content: str) -> str:
        """Detect language from file content using pattern matching."""
        scores = {}
        
        for language, patterns in cls.CONTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content, re.MULTILINE | re.IGNORECASE))
                score += matches
            scores[language] = score
        
        if scores:
            best_language = max(scores, key=scores.get)
            if scores[best_language] > 0:
                return best_language
        
        return 'unknown'
    
    @classmethod
    def _matches_patterns(cls, content: str, language: str) -> bool:
        """Check if content matches patterns for a specific language."""
        if language not in cls.CONTENT_PATTERNS:
            return False
        
        patterns = cls.CONTENT_PATTERNS[language]
        for pattern in patterns:
            if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                return True
        return False


class CPGParser(ABC):
    """Abstract base class for language-specific CPG parsers."""
    
    @abstractmethod
    def parse(self, content: str, file_path: str) -> CodePropertyGraph:
        """Parse source code and generate CPG."""
        pass
    
    @abstractmethod
    def build_ast(self, content: str, file_path: str) -> CodePropertyGraph:
        """Build Abstract Syntax Tree."""
        pass
    
    @abstractmethod
    def build_cfg(self, cpg: CodePropertyGraph) -> CodePropertyGraph:
        """Build Control Flow Graph from AST."""
        pass
    
    @abstractmethod
    def build_dfg(self, cpg: CodePropertyGraph) -> CodePropertyGraph:
        """Build Data Flow Graph from AST and CFG."""
        pass
    
    def create_node(self, node_type: NodeType, name: str, code: str = "", 
                   file_path: str = "", start_line: int = 0, end_line: int = 0,
                   start_column: int = 0, end_column: int = 0, language: str = "",
                   **properties) -> CPGNode:
        """Helper method to create CPG nodes."""
        # Convert any lists in properties to tuples to make them hashable
        safe_properties = {}
        for key, value in properties.items():
            if isinstance(value, list):
                safe_properties[key] = tuple(value)
            elif isinstance(value, set):
                safe_properties[key] = tuple(sorted(value))
            elif isinstance(value, dict):
                # For nested dictionaries, convert any list values
                safe_dict = {}
                for k, v in value.items():
                    if isinstance(v, list):
                        safe_dict[k] = tuple(v)
                    elif isinstance(v, set):
                        safe_dict[k] = tuple(sorted(v))
                    else:
                        safe_dict[k] = v
                safe_properties[key] = safe_dict
            else:
                safe_properties[key] = value
        
        return CPGNode(
            id="",  # Will be auto-generated
            node_type=node_type,
            name=name,
            code=code,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            start_column=start_column,
            end_column=end_column,
            language=language,
            properties=safe_properties
        )
    
    def create_edge(self, source_id: str, target_id: str, edge_type: EdgeType,
                   **properties) -> CPGEdge:
        """Helper method to create CPG edges."""
        # Convert any lists in properties to tuples to make them hashable
        safe_properties = {}
        for key, value in properties.items():
            if isinstance(value, list):
                safe_properties[key] = tuple(value)
            elif isinstance(value, set):
                safe_properties[key] = tuple(sorted(value))
            else:
                safe_properties[key] = value
        
        return CPGEdge(
            id="",  # Will be auto-generated
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            properties=safe_properties
        )


class CPGBuilder:
    """Main CPG builder that orchestrates the parsing process."""
    
    def __init__(self, enhanced: bool = True):
        """Initialize the CPG builder."""
        self.parsers = {}
        self.enhanced = enhanced
        self._register_parsers()
    
    def _register_parsers(self):
        """Register robust language-specific parsers."""
        # Import robust parsers here to avoid circular imports
        from parsers.robust_python_parser import RobustPythonCPGParser
        from parsers.robust_javascript_parser import RobustJavaScriptCPGParser
        from parsers.robust_java_parser import RobustJavaCPGParser
        from parsers.robust_csharp_parser import RobustCSharpCPGParser
        from parsers.robust_c_parser import RobustCCPGParser
        from parsers.robust_go_parser import RobustGoCPGParser
        from parsers.robust_php_parser import RobustPHPCPGParser
        from parsers.robust_rust_parser import RobustRustCPGParser
        from parsers.robust_ruby_parser import RobustRubyCPGParser
        from parsers.generic_parser import GenericCPGParser
        
        self.parsers = {
            'python': RobustPythonCPGParser(),
            'javascript': RobustJavaScriptCPGParser(),
            'typescript': RobustJavaScriptCPGParser(),
            'react': RobustJavaScriptCPGParser(),
            'react-ts': RobustJavaScriptCPGParser(),
            'java': RobustJavaCPGParser(),
            'cpp': RobustCCPGParser(),
            'c': RobustCCPGParser(),
            'csharp': RobustCSharpCPGParser(),
            'go': RobustGoCPGParser(),
            'php': RobustPHPCPGParser(),
            'rust': RobustRustCPGParser(),
            'ruby': RobustRubyCPGParser(),
            'generic': GenericCPGParser(),
        }
    
    def build_cpg(self, file_path: str, content: str = None) -> CodePropertyGraph:
        """Build CPG for a single file."""
        if content is None:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                except Exception as e:
                    logger.error(f"Could not read file {file_path}: {e}")
                    return CodePropertyGraph()
            except Exception as e:
                logger.error(f"Could not read file {file_path}: {e}")
                return CodePropertyGraph()
        
        # Detect language
        language = LanguageDetector.detect_language(file_path, content)
        
        # Get appropriate parser
        parser = self.parsers.get(language)
        if not parser:
            # Fallback to generic parser
            from parsers.generic_parser import GenericCPGParser
            parser = GenericCPGParser()
        
        try:
            cpg = parser.parse(content, file_path)
            cpg.metadata['language'] = language
            cpg.metadata['file_path'] = file_path
            cpg.metadata['file_size'] = len(content)
            cpg.metadata['lines'] = len(content.splitlines())
            
            logger.info(f"Built CPG for {file_path} ({language}): {len(cpg.nodes)} nodes, {len(cpg.edges)} edges")
            return cpg
            
        except Exception as e:
            logger.exception(f"Error building CPG for {file_path}: {e}")
            # Return empty CPG with metadata
            cpg = CodePropertyGraph()
            cpg.metadata['language'] = language
            cpg.metadata['file_path'] = file_path
            cpg.metadata['error'] = str(e)
            return cpg
    
    def build_cpg_batch(self, file_paths: List[str], output_dir: str = None) -> Dict[str, CodePropertyGraph]:
        """Build CPGs for multiple files with batch processing."""
        results = {}
        total_files = len(file_paths)
        
        logger.info(f"Starting batch CPG generation for {total_files} files")
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                logger.info(f"Processing file {i}/{total_files}: {file_path}")
                cpg = self.build_cpg(file_path)
                results[file_path] = cpg
                
                # Save individual CPG if output directory specified
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    file_name = Path(file_path).name + ".cpg.json"
                    cpg_file = output_path / file_name
                    self.save_cpg(cpg, str(cpg_file))
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                # Create empty CPG with error info
                cpg = CodePropertyGraph()
                cpg.metadata['file_path'] = file_path
                cpg.metadata['error'] = str(e)
                results[file_path] = cpg
        
        logger.info(f"Completed batch processing: {len(results)} files processed")
        return results
    
    def save_cpg(self, cpg: CodePropertyGraph, output_file: str):
        """Save CPG to JSON file."""
        try:
            # Convert CPG to serializable format
            cpg_data = {
                'nodes': {node_id: asdict(node) for node_id, node in cpg.nodes.items()},
                'edges': {edge_id: asdict(edge) for edge_id, edge in cpg.edges.items()},
                'metadata': cpg.metadata
            }
            
            # Convert enums to strings
            for node_data in cpg_data['nodes'].values():
                node_data['node_type'] = node_data['node_type'].value
            
            for edge_data in cpg_data['edges'].values():
                edge_data['edge_type'] = edge_data['edge_type'].value
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cpg_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved CPG to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save CPG: {e}")
    
    def load_cpg(self, input_file: str) -> CodePropertyGraph:
        """Load CPG from JSON file."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                cpg_data = json.load(f)
            
            cpg = CodePropertyGraph()
            
            # Load nodes
            for node_id, node_data in cpg_data['nodes'].items():
                node_data['node_type'] = NodeType(node_data['node_type'])
                node = CPGNode(**node_data)
                cpg.nodes[node_id] = node
            
            # Load edges
            for edge_id, edge_data in cpg_data['edges'].items():
                edge_data['edge_type'] = EdgeType(edge_data['edge_type'])
                edge = CPGEdge(**edge_data)
                cpg.edges[edge_id] = edge
            
            # Load metadata
            cpg.metadata = cpg_data.get('metadata', {})
            
            logger.info(f"Loaded CPG from {input_file}")
            return cpg
            
        except Exception as e:
            logger.error(f"Failed to load CPG: {e}")
            return CodePropertyGraph()


def get_supported_languages() -> Dict[str, str]:
    """Get dictionary of supported programming languages."""
    languages = list(set(LanguageDetector.LANGUAGE_EXTENSIONS.values()))
    return {lang: lang.title() for lang in languages}


def get_file_extensions() -> Dict[str, str]:
    """Get mapping of file extensions to languages."""
    return LanguageDetector.LANGUAGE_EXTENSIONS.copy()
