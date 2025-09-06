#!/usr/bin/env python3
"""
Robust PHP CPG Parser
=====================

Robust PHP parser with multiple parsing strategies for maximum accuracy.
"""

import re
import logging
from pathlib import Path
from parsers.robust_parser_base import RobustParserBase
from cpg_core import NodeType, EdgeType

logger = logging.getLogger(__name__)


class RobustPHPCPGParser(RobustParserBase):
    """Robust PHP CPG parser with multiple strategies."""
    
    def __init__(self):
        super().__init__()
        self.language = "php"
    
    def _transform_content(self, content: str) -> str:
        """Transform PHP content for better parsing."""
        # Remove PHP tags temporarily
        content = re.sub(r'<\?php\s*', '', content)
        content = re.sub(r'\?>', '', content)
        return content
    
    def _enhanced_regex_parsing(self, content: str):
        """Enhanced PHP regex parsing."""
        lines = content.splitlines()
        module_node = self._create_node(
            NodeType.MODULE, Path(self.current_file_path).stem,
            content[:200] + "..." if len(content) > 200 else content,
            self.current_file_path, 1, len(lines), 0, 0
        )
        
        patterns = {
            'class': r'^\s*(?:abstract\s+|final\s+)?class\s+(\w+)',
            'interface': r'^\s*interface\s+(\w+)',
            'trait': r'^\s*trait\s+(\w+)',
            'function': r'^\s*(?:public\s+|private\s+|protected\s+)?(?:static\s+)?function\s+(\w+)\s*\(',
            'variable': r'^\s*(?:public\s+|private\s+|protected\s+)?(?:static\s+)?\$(\w+)',
            'namespace': r'^\s*namespace\s+([\\w\\]+)',
            'use': r'^\s*use\s+([\\w\\]+)',
        }
        
        for i, line in enumerate(lines, 1):
            # Namespace
            match = re.match(patterns['namespace'], line)
            if match:
                ns_name = match.group(1)
                node = self._create_node(NodeType.MODULE, ns_name, line.strip(), self.current_file_path, i, i, 0, 0)
                self._create_edge(module_node, node, EdgeType.AST_CHILD)
                continue
            
            # Classes, interfaces, traits
            for pattern_type in ['class', 'interface', 'trait']:
                match = re.match(patterns[pattern_type], line)
                if match:
                    name = match.group(1)
                    node = self._create_node(NodeType.CLASS, name, line.strip(), self.current_file_path, i, i, 0, 0)
                    self._create_edge(module_node, node, EdgeType.AST_CHILD)
                    break
            
            # Functions
            match = re.match(patterns['function'], line)
            if match:
                func_name = match.group(1)
                node = self._create_node(NodeType.FUNCTION, func_name, line.strip(), self.current_file_path, i, i, 0, 0)
                self._create_edge(module_node, node, EdgeType.AST_CHILD)
            
            # Variables
            match = re.match(patterns['variable'], line)
            if match:
                var_name = match.group(1)
                node = self._create_node(NodeType.VARIABLE, var_name, line.strip(), self.current_file_path, i, i, 0, 0)
                self._create_edge(module_node, node, EdgeType.AST_CHILD)
            
            # Use statements
            match = re.match(patterns['use'], line)
            if match:
                use_name = match.group(1)
                node = self._create_node(NodeType.VARIABLE, use_name, line.strip(), self.current_file_path, i, i, 0, 0)
                self._create_edge(module_node, node, EdgeType.AST_CHILD)