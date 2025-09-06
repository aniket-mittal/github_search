#!/usr/bin/env python3
"""
Robust C# CPG Parser
=====================

Robust C# parser with multiple parsing strategies for maximum accuracy.
"""

import re
import logging
from pathlib import Path
from parsers.robust_parser_base import RobustParserBase
from cpg_core import NodeType, EdgeType

logger = logging.getLogger(__name__)


class RobustCSharpCPGParser(RobustParserBase):
    """Robust C# CPG parser with multiple strategies."""
    
    def __init__(self):
        super().__init__()
        self.language = "csharp"
    
    def _transform_content(self, content: str) -> str:
        """Transform C# content for better parsing."""
        # Remove generics and nullable types temporarily
        content = re.sub(r'<[^<>]*>', '', content)
        content = re.sub(r'\?\s*(?=[,\)\];])', '', content)  # Remove nullable ?
        return content
    
    def _enhanced_regex_parsing(self, content: str):
        """Enhanced C# regex parsing."""
        lines = content.splitlines()
        module_node = self._create_node(
            NodeType.MODULE, Path(self.current_file_path).stem,
            content[:200] + "..." if len(content) > 200 else content,
            self.current_file_path, 1, len(lines), 0, 0
        )
        
        patterns = {
            'namespace': r'^\s*namespace\s+([\w\.]+)',
            'class': r'^\s*(?:public|private|protected|internal)?\s*(?:abstract|sealed|static|partial)?\s*class\s+(\w+)',
            'interface': r'^\s*(?:public|private|protected|internal)?\s*interface\s+(\w+)',
            'struct': r'^\s*(?:public|private|protected|internal)?\s*struct\s+(\w+)',
            'enum': r'^\s*(?:public|private|protected|internal)?\s*enum\s+(\w+)',
            'method': r'^\s*(?:public|private|protected|internal)?\s*(?:static)?\s*(?:virtual|override|abstract)?\s*\w+\s+(\w+)\s*\([^)]*\)\s*\{?',
            'property': r'^\s*(?:public|private|protected|internal)?\s*\w+\s+(\w+)\s*\{',
            'field': r'^\s*(?:public|private|protected|internal)?\s*(?:static|readonly)?\s*\w+\s+(\w+)(?:\s*=.*)?;',
            'using': r'^\s*using\s+([\w\.]+);',
        }
        
        for i, line in enumerate(lines, 1):
            # Namespace
            match = re.match(patterns['namespace'], line)
            if match:
                ns_name = match.group(1)
                node = self._create_node(NodeType.MODULE, ns_name, line.strip(), self.current_file_path, i, i, 0, 0)
                self._create_edge(module_node, node, EdgeType.AST_CHILD)
                continue
            
            # Classes, interfaces, structs, enums
            for pattern_type in ['class', 'interface', 'struct', 'enum']:
                match = re.match(patterns[pattern_type], line)
                if match:
                    name = match.group(1)
                    node = self._create_node(NodeType.CLASS, name, line.strip(), self.current_file_path, i, i, 0, 0)
                    self._create_edge(module_node, node, EdgeType.AST_CHILD)
                    break
            
            # Methods
            match = re.match(patterns['method'], line)
            if match and 'class' not in line and 'interface' not in line:
                method_name = match.group(1)
                if method_name not in ['class', 'interface', 'struct', 'enum', 'get', 'set']:
                    node = self._create_node(NodeType.FUNCTION, method_name, line.strip(), self.current_file_path, i, i, 0, 0)
                    self._create_edge(module_node, node, EdgeType.AST_CHILD)
            
            # Properties  
            match = re.match(patterns['property'], line)
            if match:
                prop_name = match.group(1)
                node = self._create_node(NodeType.VARIABLE, prop_name, line.strip(), self.current_file_path, i, i, 0, 0)
                self._create_edge(module_node, node, EdgeType.AST_CHILD)
            
            # Fields
            match = re.match(patterns['field'], line)
            if match:
                field_name = match.group(1)
                node = self._create_node(NodeType.VARIABLE, field_name, line.strip(), self.current_file_path, i, i, 0, 0)
                self._create_edge(module_node, node, EdgeType.AST_CHILD)
            
            # Using statements
            match = re.match(patterns['using'], line)
            if match:
                using_name = match.group(1)
                node = self._create_node(NodeType.VARIABLE, using_name, line.strip(), self.current_file_path, i, i, 0, 0)
                self._create_edge(module_node, node, EdgeType.AST_CHILD)