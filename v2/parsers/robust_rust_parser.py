#!/usr/bin/env python3
"""
Robust Rust CPG Parser
=======================

Robust Rust parser with multiple parsing strategies for maximum accuracy.
"""

import re
import logging
from pathlib import Path
from parsers.robust_parser_base import RobustParserBase
from cpg_core import NodeType, EdgeType

logger = logging.getLogger(__name__)


class RobustRustCPGParser(RobustParserBase):
    """Robust Rust CPG parser with multiple strategies."""
    
    def __init__(self):
        super().__init__()
        self.language = "rust"
    
    def _transform_content(self, content: str) -> str:
        """Transform Rust content for better parsing."""
        # Remove lifetimes temporarily
        content = re.sub(r"'[a-zA-Z_]\w*", '', content)
        # Remove generics temporarily
        content = re.sub(r'<[^<>]*>', '', content)
        return content
    
    def _enhanced_regex_parsing(self, content: str):
        """Enhanced Rust regex parsing."""
        lines = content.splitlines()
        module_node = self._create_node(
            NodeType.MODULE, Path(self.current_file_path).stem,
            content[:200] + "..." if len(content) > 200 else content,
            self.current_file_path, 1, len(lines), 0, 0
        )
        
        patterns = {
            'struct': r'^\s*(?:pub\s+)?struct\s+(\w+)',
            'enum': r'^\s*(?:pub\s+)?enum\s+(\w+)',
            'trait': r'^\s*(?:pub\s+)?trait\s+(\w+)',
            'impl': r'^\s*impl(?:\s+\w+)?\s+(?:for\s+)?(\w+)',
            'function': r'^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*\(',
            'const': r'^\s*(?:pub\s+)?const\s+(\w+)',
            'static': r'^\s*(?:pub\s+)?static\s+(\w+)',
            'mod': r'^\s*(?:pub\s+)?mod\s+(\w+)',
            'use': r'^\s*use\s+([:\w]+)',
            'macro': r'^\s*macro_rules!\s+(\w+)',
        }
        
        for i, line in enumerate(lines, 1):
            # Structs, enums, traits
            for pattern_type in ['struct', 'enum', 'trait']:
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
            
            # Constants and statics
            for pattern_type in ['const', 'static']:
                match = re.match(patterns[pattern_type], line)
                if match:
                    name = match.group(1)
                    node = self._create_node(NodeType.VARIABLE, name, line.strip(), self.current_file_path, i, i, 0, 0)
                    self._create_edge(module_node, node, EdgeType.AST_CHILD)
            
            # Modules
            match = re.match(patterns['mod'], line)
            if match:
                mod_name = match.group(1)
                node = self._create_node(NodeType.MODULE, mod_name, line.strip(), self.current_file_path, i, i, 0, 0)
                self._create_edge(module_node, node, EdgeType.AST_CHILD)
            
            # Use statements
            match = re.match(patterns['use'], line)
            if match:
                use_name = match.group(1)
                node = self._create_node(NodeType.VARIABLE, use_name, line.strip(), self.current_file_path, i, i, 0, 0)
                self._create_edge(module_node, node, EdgeType.AST_CHILD)
            
            # Macros
            match = re.match(patterns['macro'], line)
            if match:
                macro_name = match.group(1)
                node = self._create_node(NodeType.FUNCTION, macro_name, line.strip(), self.current_file_path, i, i, 0, 0)
                self._create_edge(module_node, node, EdgeType.AST_CHILD)