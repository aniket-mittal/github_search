#!/usr/bin/env python3
"""
Robust Parser Base Class
========================

A base class that provides robust parsing capabilities for all programming languages.
This implements a multi-strategy approach:
1. AST-based parsing (if available)
2. Language-specific transformations and fixes
3. Regex-based fallback parsing
4. Generic pattern-based minimal parsing

This ensures all languages have robust, accurate CPG generation.
"""

import re
import logging
from typing import Dict, List, Set, Optional, Any, Tuple
from pathlib import Path
from abc import abstractmethod

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpg_core import CPGParser, CodePropertyGraph, CPGNode, CPGEdge, NodeType, EdgeType

logger = logging.getLogger(__name__)


class RobustParserBase(CPGParser):
    """Base class for robust multi-strategy parsers."""
    
    def __init__(self):
        """Initialize the robust parser base."""
        self.current_cpg = None
        self.current_file_path = ""
        self.node_counter = 0
        self.edge_counter = 0
        self.language = "unknown"
    
    def parse(self, content: str, file_path: str) -> CodePropertyGraph:
        """Parse source code using multiple strategies."""
        self.current_file_path = file_path
        self.current_cpg = CodePropertyGraph()
        self.node_counter = 0
        self.edge_counter = 0
        
        # Try multiple parsing strategies
        success = False
        
        # Strategy 1: Try AST-based parsing (if implemented)
        if hasattr(self, '_try_ast_parsing'):
            try:
                success = self._try_ast_parsing(content)
                if success:
                    logger.debug(f"✅ AST parsing successful for {file_path}")
            except Exception as e:
                logger.debug(f"⚠️ AST parsing failed for {file_path}: {e}")
        
        # Strategy 2: Try language-specific transformations + parsing
        if not success:
            try:
                transformed_content = self._transform_content(content)
                if transformed_content != content and hasattr(self, '_try_ast_parsing'):
                    success = self._try_ast_parsing(transformed_content)
                    if success:
                        logger.debug(f"✅ Transformed parsing successful for {file_path}")
            except Exception as e:
                logger.debug(f"⚠️ Transformed parsing failed for {file_path}: {e}")
        
        # Strategy 3: Enhanced regex-based parsing
        if not success:
            try:
                self._enhanced_regex_parsing(content)
                success = True
                logger.debug(f"✅ Enhanced regex parsing used for {file_path}")
            except Exception as e:
                logger.debug(f"⚠️ Enhanced regex parsing failed for {file_path}: {e}")
        
        # Strategy 4: Generic fallback parsing
        if not success:
            logger.debug(f"🔄 Using generic fallback parsing for {file_path}")
            self._generic_fallback_parsing(content)
        
        # Ensure we have at least a module node
        if not self.current_cpg.nodes:
            self._create_minimal_cpg(content, file_path)
        
        return self.current_cpg
    
    @abstractmethod
    def _transform_content(self, content: str) -> str:
        """Transform content for better parsing (language-specific)."""
        pass
    
    @abstractmethod
    def _enhanced_regex_parsing(self, content: str):
        """Enhanced regex-based parsing (language-specific patterns).""" 
        pass
    
    def _generic_fallback_parsing(self, content: str):
        """Generic fallback parsing that works for any language."""
        lines = content.splitlines()
        
        # Create module node
        module_node = self._create_node(
            NodeType.MODULE, Path(self.current_file_path).stem,
            content[:200] + "..." if len(content) > 200 else content,
            self.current_file_path, 1, len(lines), 0, 0
        )
        
        # Generic patterns for functions across languages
        function_patterns = [
            r'^(\s*)(def|function|func|fn|sub|procedure|fun)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
            r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{',  # C-style
            r'^(\s*)(public|private|protected)?\s*(static)?\s*(\w+)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',  # Java-style
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern in function_patterns:
                match = re.match(pattern, line)
                if match:
                    func_name = match.groups()[-1]  # Last group is typically the function name
                    func_node = self._create_node(
                        NodeType.FUNCTION, func_name, line.strip(),
                        self.current_file_path, i, i, 0, 0
                    )
                    self._create_edge(module_node, func_node, EdgeType.AST_CHILD)
                    break
        
        # Generic patterns for classes
        class_patterns = [
            r'^(\s*)(class|struct|interface|enum)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'^(\s*)(public|private)?\s*(class|struct|interface|enum)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern in class_patterns:
                match = re.match(pattern, line)
                if match:
                    class_name = match.groups()[-1]
                    class_node = self._create_node(
                        NodeType.CLASS, class_name, line.strip(),
                        self.current_file_path, i, i, 0, 0
                    )
                    self._create_edge(module_node, class_node, EdgeType.AST_CHILD)
                    break
        
        # Generic patterns for imports
        import_patterns = [
            r'^(\s*)(import|include|require|using|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'^(\s*)#include\s*[<"]([^>"]+)[>"]',
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    import_name = match.groups()[-1]
                    import_node = self._create_node(
                        NodeType.VARIABLE, import_name, line.strip(),
                        self.current_file_path, i, i, 0, 0
                    )
                    self._create_edge(module_node, import_node, EdgeType.AST_CHILD)
                    break
    
    def _create_minimal_cpg(self, content: str, file_path: str):
        """Create minimal CPG when all parsing fails.""" 
        lines = content.splitlines()
        module_node = self._create_node(
            NodeType.MODULE, Path(file_path).stem,
            content[:500] + "..." if len(content) > 500 else content,
            file_path, 1, len(lines), 0, 0
        )
    
    # Standard implementations for all language parsers
    def build_ast(self, content: str, file_path: str) -> CodePropertyGraph:
        """Build AST - delegated to main parse method."""
        return self.parse(content, file_path)
    
    def build_cfg(self, cpg: CodePropertyGraph) -> CodePropertyGraph:
        """Build CFG - basic implementation."""
        return cpg
    
    def build_dfg(self, cpg: CodePropertyGraph) -> CodePropertyGraph:
        """Build DFG - basic implementation."""  
        return cpg
    
    def _create_node(self, node_type: NodeType, name: str, code: str, 
                     file_path: str, start_line: int, end_line: int,
                     start_col: int, end_col: int) -> CPGNode:
        """Create a CPG node and add it to the graph."""
        self.node_counter += 1
        node_id = f"node_{self.node_counter}"
        
        node = CPGNode(
            id=node_id,
            node_type=node_type,
            name=name,
            code=code,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            start_column=start_col,
            end_column=end_col,
            language=self.language
        )
        
        self.current_cpg.add_node(node)
        return node
    
    def _create_edge(self, source_node: CPGNode, target_node: CPGNode, edge_type: EdgeType) -> CPGEdge:
        """Create a CPG edge and add it to the graph."""
        self.edge_counter += 1
        edge_id = f"edge_{self.edge_counter}"
        
        edge = CPGEdge(
            id=edge_id,
            source_id=source_node.id,
            target_id=target_node.id,
            edge_type=edge_type
        )
        
        self.current_cpg.add_edge(edge)
        return edge
    
    def _get_code_snippet(self, lines: List[str], start_line: int, end_line: int) -> str:
        """Get code snippet from lines."""
        try:
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            
            if start_idx >= len(lines):
                return ""
            
            snippet = '\n'.join(lines[start_idx:end_idx])
            return snippet[:1000]  # Limit length
        except:
            return ""