#!/usr/bin/env python3
"""
Robust Java CPG Parser
=======================

Robust Java parser with multiple parsing strategies for maximum accuracy.
"""

import re
import logging
from pathlib import Path
from parsers.robust_parser_base import RobustParserBase
from cpg_core import NodeType, EdgeType

logger = logging.getLogger(__name__)


class RobustJavaCPGParser(RobustParserBase):
    """Robust Java CPG parser with multiple strategies."""
    
    def __init__(self):
        super().__init__()
        self.language = "java"
    
    def _transform_content(self, content: str) -> str:
        """Transform Java content for better parsing."""
        # Remove generics temporarily for parsing
        content = re.sub(r'<[^<>]*>', '', content)
        # Normalize access modifiers
        content = re.sub(r'\b(public|private|protected)\s+', r'\1 ', content)
        return content
    
    def _enhanced_regex_parsing(self, content: str):
        """Enhanced Java regex parsing."""
        lines = content.splitlines()
        module_node = self._create_node(
            NodeType.MODULE, Path(self.current_file_path).stem,
            content[:200] + "..." if len(content) > 200 else content,
            self.current_file_path, 1, len(lines), 0, 0
        )
        
        patterns = {
            'class': r'^\s*(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+(\w+)',
            'interface': r'^\s*(?:public|private|protected)?\s*interface\s+(\w+)',
            'method': r'^\s*(?:public|private|protected)?\s*(?:static)?\s*(?:final)?\s*\w+\s+(\w+)\s*\([^)]*\)\s*\{?',
            'field': r'^\s*(?:public|private|protected)?\s*(?:static)?\s*(?:final)?\s*\w+\s+(\w+)(?:\s*=.*)?;',
            'import': r'^\s*import\s+(?:static\s+)?([\w\.]+);',
            'package': r'^\s*package\s+([\w\.]+);',
        }
        
        for i, line in enumerate(lines, 1):
            # Classes and interfaces
            for pattern_type in ['class', 'interface']:
                match = re.match(patterns[pattern_type], line)
                if match:
                    name = match.group(1)
                    node_type = NodeType.CLASS if pattern_type == 'class' else NodeType.CLASS
                    node = self._create_node(node_type, name, line.strip(), self.current_file_path, i, i, 0, 0)
                    self._create_edge(module_node, node, EdgeType.AST_CHILD)
                    break
            
            # Methods
            match = re.match(patterns['method'], line)
            if match and 'class' not in line and 'interface' not in line:
                method_name = match.group(1)
                if method_name not in ['class', 'interface', 'extends', 'implements']:
                    node = self._create_node(NodeType.FUNCTION, method_name, line.strip(), self.current_file_path, i, i, 0, 0)
                    self._create_edge(module_node, node, EdgeType.AST_CHILD)
            
            # Fields
            match = re.match(patterns['field'], line)
            if match:
                field_name = match.group(1)
                node = self._create_node(NodeType.VARIABLE, field_name, line.strip(), self.current_file_path, i, i, 0, 0)
                self._create_edge(module_node, node, EdgeType.AST_CHILD)
            
            # Imports
            match = re.match(patterns['import'], line)
            if match:
                import_name = match.group(1)
                node = self._create_node(NodeType.VARIABLE, import_name, line.strip(), self.current_file_path, i, i, 0, 0)
                self._create_edge(module_node, node, EdgeType.AST_CHILD)