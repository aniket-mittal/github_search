#!/usr/bin/env python3
"""
Robust C/C++ CPG Parser
========================

Robust C/C++ parser with multiple parsing strategies for maximum accuracy.
"""

import re
import logging
from pathlib import Path
from parsers.robust_parser_base import RobustParserBase
from cpg_core import NodeType, EdgeType

logger = logging.getLogger(__name__)


class RobustCCPGParser(RobustParserBase):
    """Robust C/C++ CPG parser with multiple strategies."""
    
    def __init__(self):
        super().__init__()
        self.language = "c"
    
    def _transform_content(self, content: str) -> str:
        """Transform C/C++ content for better parsing."""
        # Remove preprocessor directives temporarily
        content = re.sub(r'^\s*#.*$', '', content, flags=re.MULTILINE)
        # Remove template syntax temporarily
        content = re.sub(r'template\s*<[^>]*>', '', content)
        # Normalize pointer syntax
        content = re.sub(r'\*\s*(\w+)', r'* \1', content)
        return content
    
    def _enhanced_regex_parsing(self, content: str):
        """Enhanced C/C++ regex parsing."""
        lines = content.splitlines()
        module_node = self._create_node(
            NodeType.MODULE, Path(self.current_file_path).stem,
            content[:200] + "..." if len(content) > 200 else content,
            self.current_file_path, 1, len(lines), 0, 0
        )
        
        patterns = {
            'struct': r'^\s*(?:typedef\s+)?struct\s+(\w+)',
            'class': r'^\s*class\s+(\w+)',
            'function': r'^\s*(?:static\s+)?(?:inline\s+)?(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*\{',
            'typedef': r'^\s*typedef\s+.*\s+(\w+);',
            'define': r'^\s*#define\s+(\w+)',
            'include': r'^\s*#include\s*[<"]([\w./]+)[>"]',
            'variable': r'^\s*(?:static\s+)?(?:const\s+)?(?:extern\s+)?\w+\s+(\w+)(?:\s*=.*)?;',
            'enum': r'^\s*enum\s+(\w+)',
        }
        
        for i, line in enumerate(lines, 1):
            # Structs and classes
            for pattern_type in ['struct', 'class']:
                match = re.match(patterns[pattern_type], line)
                if match:
                    name = match.group(1)
                    node = self._create_node(NodeType.CLASS, name, line.strip(), self.current_file_path, i, i, 0, 0)
                    self._create_edge(module_node, node, EdgeType.AST_CHILD)
                    break
            
            # Functions
            match = re.match(patterns['function'], line)
            if match and 'struct' not in line and 'class' not in line:
                func_name = match.group(1)
                if func_name not in ['struct', 'class', 'typedef', 'enum', 'if', 'for', 'while']:
                    node = self._create_node(NodeType.FUNCTION, func_name, line.strip(), self.current_file_path, i, i, 0, 0)
                    self._create_edge(module_node, node, EdgeType.AST_CHILD)
            
            # Variables
            match = re.match(patterns['variable'], line)
            if match:
                var_name = match.group(1)
                node = self._create_node(NodeType.VARIABLE, var_name, line.strip(), self.current_file_path, i, i, 0, 0)
                self._create_edge(module_node, node, EdgeType.AST_CHILD)
            
            # Typedefs and defines
            for pattern_type in ['typedef', 'define']:
                match = re.match(patterns[pattern_type], line)
                if match:
                    name = match.group(1)
                    node = self._create_node(NodeType.VARIABLE, name, line.strip(), self.current_file_path, i, i, 0, 0)
                    self._create_edge(module_node, node, EdgeType.AST_CHILD)
            
            # Includes
            match = re.match(patterns['include'], line)
            if match:
                include_name = match.group(1)
                node = self._create_node(NodeType.VARIABLE, include_name, line.strip(), self.current_file_path, i, i, 0, 0)
                self._create_edge(module_node, node, EdgeType.AST_CHILD)
            
            # Enums
            match = re.match(patterns['enum'], line)
            if match:
                enum_name = match.group(1)
                node = self._create_node(NodeType.CLASS, enum_name, line.strip(), self.current_file_path, i, i, 0, 0)
                self._create_edge(module_node, node, EdgeType.AST_CHILD)