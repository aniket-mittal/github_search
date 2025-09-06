#!/usr/bin/env python3
"""
Robust Ruby CPG Parser
=======================

Robust Ruby parser with multiple parsing strategies for maximum accuracy.
"""

import re
import logging
from pathlib import Path
from parsers.robust_parser_base import RobustParserBase
from cpg_core import NodeType, EdgeType

logger = logging.getLogger(__name__)


class RobustRubyCPGParser(RobustParserBase):
    """Robust Ruby CPG parser with multiple strategies."""
    
    def __init__(self):
        super().__init__()
        self.language = "ruby"
    
    def _transform_content(self, content: str) -> str:
        """Transform Ruby content for better parsing."""
        # Normalize method definitions
        content = re.sub(r'\bdef\s+', 'def ', content)
        return content
    
    def _enhanced_regex_parsing(self, content: str):
        """Enhanced Ruby regex parsing."""
        lines = content.splitlines()
        module_node = self._create_node(
            NodeType.MODULE, Path(self.current_file_path).stem,
            content[:200] + "..." if len(content) > 200 else content,
            self.current_file_path, 1, len(lines), 0, 0
        )
        
        patterns = {
            'class': r'^\s*class\s+(\w+)',
            'module': r'^\s*module\s+(\w+)',
            'method': r'^\s*def\s+(\w+)',
            'attr': r'^\s*attr_(?:reader|writer|accessor)\s+:(\w+)',
            'require': r'^\s*require\s+["\']([^"\']+)["\']',
            'include': r'^\s*include\s+(\w+)',
            'extend': r'^\s*extend\s+(\w+)',
        }
        
        for i, line in enumerate(lines, 1):
            # Classes and modules
            for pattern_type in ['class', 'module']:
                match = re.match(patterns[pattern_type], line)
                if match:
                    name = match.group(1)
                    node_type = NodeType.CLASS if pattern_type == 'class' else NodeType.MODULE
                    node = self._create_node(node_type, name, line.strip(), self.current_file_path, i, i, 0, 0)
                    self._create_edge(module_node, node, EdgeType.AST_CHILD)
                    break
            
            # Methods
            match = re.match(patterns['method'], line)
            if match:
                method_name = match.group(1)
                node = self._create_node(NodeType.FUNCTION, method_name, line.strip(), self.current_file_path, i, i, 0, 0)
                self._create_edge(module_node, node, EdgeType.AST_CHILD)
            
            # Attributes
            match = re.match(patterns['attr'], line)
            if match:
                attr_name = match.group(1)
                node = self._create_node(NodeType.VARIABLE, attr_name, line.strip(), self.current_file_path, i, i, 0, 0)
                self._create_edge(module_node, node, EdgeType.AST_CHILD)
            
            # Requires, includes, extends
            for pattern_type in ['require', 'include', 'extend']:
                match = re.match(patterns[pattern_type], line)
                if match:
                    name = match.group(1)
                    node = self._create_node(NodeType.VARIABLE, name, line.strip(), self.current_file_path, i, i, 0, 0)
                    self._create_edge(module_node, node, EdgeType.AST_CHILD)