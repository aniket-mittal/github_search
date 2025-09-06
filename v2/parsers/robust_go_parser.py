#!/usr/bin/env python3
"""
Robust Go CPG Parser
====================

Robust Go parser with multiple parsing strategies for maximum accuracy.
"""

import re
import logging
from pathlib import Path
from parsers.robust_parser_base import RobustParserBase
from cpg_core import NodeType, EdgeType

logger = logging.getLogger(__name__)


class RobustGoCPGParser(RobustParserBase):
    """Robust Go CPG parser with multiple strategies."""
    
    def __init__(self):
        super().__init__()
        self.language = "go"
    
    def _transform_content(self, content: str) -> str:
        """Transform Go content for better parsing."""
        # Normalize function syntax
        content = re.sub(r'\bfunc\s+', 'func ', content)
        # Normalize struct syntax
        content = re.sub(r'\bstruct\s+', 'struct ', content)
        return content
    
    def _enhanced_regex_parsing(self, content: str):
        """Enhanced Go regex parsing."""
        lines = content.splitlines()
        module_node = self._create_node(
            NodeType.MODULE, Path(self.current_file_path).stem,
            content[:200] + "..." if len(content) > 200 else content,
            self.current_file_path, 1, len(lines), 0, 0
        )
        
        patterns = {
            'package': r'^\s*package\s+(\w+)',
            'import': r'^\s*import\s+(?:"([^"]+)"|`([^`]+)`)',
            'struct': r'^\s*type\s+(\w+)\s+struct',
            'interface': r'^\s*type\s+(\w+)\s+interface',
            'function': r'^\s*func\s+(?:\([^)]*\)\s+)?(\w+)\s*\([^)]*\)',
            'method': r'^\s*func\s+\([^)]*\)\s+(\w+)\s*\([^)]*\)',
            'variable': r'^\s*(?:var\s+(\w+)|(\w+)\s*:=)',
            'const': r'^\s*const\s+(\w+)',
            'type': r'^\s*type\s+(\w+)\s+\w+',
        }
        
        for i, line in enumerate(lines, 1):
            # Package declaration
            match = re.match(patterns['package'], line)
            if match:
                pkg_name = match.group(1)
                node = self._create_node(NodeType.MODULE, pkg_name, line.strip(), self.current_file_path, i, i, 0, 0)
                self._create_edge(module_node, node, EdgeType.AST_CHILD)
                continue
            
            # Structs and interfaces
            for pattern_type in ['struct', 'interface']:
                match = re.match(patterns[pattern_type], line)
                if match:
                    name = match.group(1)
                    node = self._create_node(NodeType.CLASS, name, line.strip(), self.current_file_path, i, i, 0, 0)
                    self._create_edge(module_node, node, EdgeType.AST_CHILD)
                    break
            
            # Functions and methods
            for pattern_type in ['function', 'method']:
                match = re.match(patterns[pattern_type], line)
                if match:
                    func_name = match.group(1)
                    node = self._create_node(NodeType.FUNCTION, func_name, line.strip(), self.current_file_path, i, i, 0, 0)
                    self._create_edge(module_node, node, EdgeType.AST_CHILD)
                    break
            
            # Variables
            match = re.match(patterns['variable'], line)
            if match:
                var_name = match.group(1) or match.group(2)
                if var_name:
                    node = self._create_node(NodeType.VARIABLE, var_name, line.strip(), self.current_file_path, i, i, 0, 0)
                    self._create_edge(module_node, node, EdgeType.AST_CHILD)
            
            # Constants and types
            for pattern_type in ['const', 'type']:
                match = re.match(patterns[pattern_type], line)
                if match:
                    name = match.group(1)
                    node = self._create_node(NodeType.VARIABLE, name, line.strip(), self.current_file_path, i, i, 0, 0)
                    self._create_edge(module_node, node, EdgeType.AST_CHILD)
            
            # Imports
            match = re.match(patterns['import'], line)
            if match:
                import_name = match.group(1) or match.group(2)
                node = self._create_node(NodeType.VARIABLE, import_name, line.strip(), self.current_file_path, i, i, 0, 0)
                self._create_edge(module_node, node, EdgeType.AST_CHILD)