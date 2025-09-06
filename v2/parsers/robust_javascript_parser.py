#!/usr/bin/env python3
"""
Robust JavaScript/TypeScript CPG Parser
========================================

A robust JavaScript/TypeScript parser that uses multiple parsing strategies:
1. Attempts AST parsing using available JS parsers (if installed)
2. Handles common JS/TS variations and transformations
3. Enhanced regex-based parsing with comprehensive patterns
4. Generic fallback parsing

This ensures accurate CPG generation for both JavaScript and TypeScript code.
"""

import re
import logging
from typing import Dict, List, Set, Optional, Any, Tuple
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsers.robust_parser_base import RobustParserBase
from cpg_core import CodePropertyGraph, CPGNode, CPGEdge, NodeType, EdgeType

logger = logging.getLogger(__name__)


class RobustJavaScriptCPGParser(RobustParserBase):
    """Robust JavaScript/TypeScript CPG parser with multiple strategies."""
    
    def __init__(self):
        """Initialize the robust JavaScript parser."""
        super().__init__()
        self.language = "javascript"
        self.variable_definitions = {}
        self.variable_uses = {}
        self.scope_stack = []
        self.current_scope = "global"
    
    def _transform_content(self, content: str) -> str:
        """Transform JS/TS content for better parsing."""
        # Remove TypeScript-specific syntax that might cause issues
        content = re.sub(r':\s*\w+(\[\])?(?=\s*[;,=\)])', '', content)  # Remove type annotations
        content = re.sub(r'<[^>]*>', '', content)  # Remove generic type parameters
        content = re.sub(r'\bas\s+\w+', '', content)  # Remove 'as' type assertions
        content = re.sub(r'\!\.', '.', content)  # Remove non-null assertions
        
        # Normalize function declarations
        content = re.sub(r'export\s+default\s+function', 'function', content)
        content = re.sub(r'export\s+function', 'function', content)
        
        return content
    
    def _try_ast_parsing(self, content: str) -> bool:
        """Try AST-based parsing for JavaScript."""
        # Try to use available JS parsers
        try:
            # Try esprima if available
            import esprima
            try:
                ast = esprima.parseScript(content, {'loc': True, 'range': True})
                self._build_cpg_from_js_ast(ast, content)
                return True
            except:
                ast = esprima.parseModule(content, {'loc': True, 'range': True})
                self._build_cpg_from_js_ast(ast, content)
                return True
        except ImportError:
            pass
        
        # Could add other JS parsers here (acorn, etc.)
        return False
    
    def _build_cpg_from_js_ast(self, ast: dict, content: str):
        """Build CPG from JavaScript AST."""
        lines = content.splitlines()
        
        # Create module node
        module_node = self._create_node(
            NodeType.MODULE, Path(self.current_file_path).stem,
            content[:200] + "..." if len(content) > 200 else content,
            self.current_file_path, 1, len(lines), 0, 0
        )
        
        # Process AST recursively
        self._process_js_ast_node(ast, module_node, lines)
    
    def _process_js_ast_node(self, node: dict, parent_node: CPGNode, lines: List[str]):
        """Process JavaScript AST node."""
        if not isinstance(node, dict) or 'type' not in node:
            return
        
        node_type = node.get('type')
        
        if node_type == 'FunctionDeclaration':
            self._process_js_function(node, parent_node, lines)
        elif node_type == 'ClassDeclaration':
            self._process_js_class(node, parent_node, lines)
        elif node_type == 'VariableDeclaration':
            self._process_js_variable(node, parent_node, lines)
        elif node_type == 'CallExpression':
            self._process_js_call(node, parent_node, lines)
        elif node_type == 'ImportDeclaration':
            self._process_js_import(node, parent_node, lines)
        
        # Process child nodes
        for key, value in node.items():
            if key == 'body' and isinstance(value, list):
                for child in value:
                    self._process_js_ast_node(child, parent_node, lines)
            elif isinstance(value, dict):
                self._process_js_ast_node(value, parent_node, lines)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._process_js_ast_node(item, parent_node, lines)
    
    def _process_js_function(self, node: dict, parent_node: CPGNode, lines: List[str]):
        """Process JavaScript function."""
        func_name = node.get('id', {}).get('name', 'anonymous')
        loc = node.get('loc', {})
        start_line = loc.get('start', {}).get('line', 1)
        end_line = loc.get('end', {}).get('line', start_line)
        
        func_code = self._get_code_snippet(lines, start_line, min(start_line + 3, len(lines)))
        
        func_node = self._create_node(
            NodeType.FUNCTION, func_name, func_code,
            self.current_file_path, start_line, end_line, 0, 0
        )
        
        self._create_edge(parent_node, func_node, EdgeType.AST_CHILD)
    
    def _process_js_class(self, node: dict, parent_node: CPGNode, lines: List[str]):
        """Process JavaScript class."""
        class_name = node.get('id', {}).get('name', 'anonymous')
        loc = node.get('loc', {})
        start_line = loc.get('start', {}).get('line', 1)
        end_line = loc.get('end', {}).get('line', start_line)
        
        class_code = self._get_code_snippet(lines, start_line, min(start_line + 2, len(lines)))
        
        class_node = self._create_node(
            NodeType.CLASS, class_name, class_code,
            self.current_file_path, start_line, end_line, 0, 0
        )
        
        self._create_edge(parent_node, class_node, EdgeType.AST_CHILD)
    
    def _process_js_variable(self, node: dict, parent_node: CPGNode, lines: List[str]):
        """Process JavaScript variable."""
        declarations = node.get('declarations', [])
        for decl in declarations:
            var_name = decl.get('id', {}).get('name')
            if var_name:
                loc = node.get('loc', {})
                start_line = loc.get('start', {}).get('line', 1)
                
                var_code = self._get_code_snippet(lines, start_line, start_line)
                
                var_node = self._create_node(
                    NodeType.VARIABLE, var_name, var_code,
                    self.current_file_path, start_line, start_line, 0, 0
                )
                
                self._create_edge(parent_node, var_node, EdgeType.AST_CHILD)
    
    def _process_js_call(self, node: dict, parent_node: CPGNode, lines: List[str]):
        """Process JavaScript function call."""
        callee = node.get('callee', {})
        call_name = "call"
        
        if callee.get('type') == 'Identifier':
            call_name = callee.get('name', 'call')
        elif callee.get('type') == 'MemberExpression':
            property_name = callee.get('property', {}).get('name')
            if property_name:
                call_name = property_name
        
        loc = node.get('loc', {})
        start_line = loc.get('start', {}).get('line', 1)
        
        call_code = self._get_code_snippet(lines, start_line, start_line)
        
        call_node = self._create_node(
            NodeType.CALL, call_name, call_code,
            self.current_file_path, start_line, start_line, 0, 0
        )
        
        self._create_edge(parent_node, call_node, EdgeType.AST_CHILD)
    
    def _process_js_import(self, node: dict, parent_node: CPGNode, lines: List[str]):
        """Process JavaScript import."""
        source = node.get('source', {}).get('value', 'unknown')
        loc = node.get('loc', {})
        start_line = loc.get('start', {}).get('line', 1)
        
        import_code = self._get_code_snippet(lines, start_line, start_line)
        
        import_node = self._create_node(
            NodeType.VARIABLE, source, import_code,
            self.current_file_path, start_line, start_line, 0, 0
        )
        
        self._create_edge(parent_node, import_node, EdgeType.AST_CHILD)
    
    def _enhanced_regex_parsing(self, content: str):
        """Enhanced regex-based parsing for JavaScript.""" 
        lines = content.splitlines()
        
        # Create module node
        module_node = self._create_node(
            NodeType.MODULE, Path(self.current_file_path).stem,
            content[:200] + "..." if len(content) > 200 else content,
            self.current_file_path, 1, len(lines), 0, 0
        )
        
        # Enhanced JavaScript patterns
        patterns = {
            'class': r'^\s*(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{',
            'function_decl': r'^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)\s*\{',
            'function_expr': r'^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function\s*\([^)]*\)\s*\{',
            'arrow_function': r'^\s*(?:const|let|var)\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*\{?',
            'method': r'^\s*(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{',
            'import': r'^\s*import\s+.*?from\s+[\'"]([^\'"]+)[\'"];?',
            'require': r'^\s*(?:const|let|var)\s+.*?require\s*\([\'"]([^\'"]+)[\'"]\)',
            'export': r'^\s*export\s+(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)',
            'variable_decl': r'^\s*(?:const|let|var)\s+(\w+)(?:\s*[:=].*?)?[;\n]?$',
            'assignment': r'^\s*(\w+(?:\.\w+)*)\s*=\s*([^;\n]+)[;\n]?$',
            'if_statement': r'^\s*if\s*\([^)]+\)\s*\{',
            'for_loop': r'^\s*for\s*\([^)]*\)\s*\{',
            'while_loop': r'^\s*while\s*\([^)]+\)\s*\{',
            'try_catch': r'^\s*try\s*\{',
        }
        
        for i, line in enumerate(lines, 1):
            # Process classes
            if re.match(patterns['class'], line):
                match = re.match(patterns['class'], line)
                if match:
                    class_name = match.group(1)
                    class_node = self._create_node(
                        NodeType.CLASS, class_name, line.strip(),
                        self.current_file_path, i, i, 0, 0
                    )
                    self._create_edge(module_node, class_node, EdgeType.AST_CHILD)
            
            # Process functions (all types)
            func_patterns = [patterns['function_decl'], patterns['function_expr'], 
                            patterns['arrow_function'], patterns['method']]
            for pattern in func_patterns:
                match = re.match(pattern, line)
                if match:
                    func_name = match.group(1)
                    func_node = self._create_node(
                        NodeType.FUNCTION, func_name, line.strip(),
                        self.current_file_path, i, i, 0, 0
                    )
                    self._create_edge(module_node, func_node, EdgeType.AST_CHILD)
                    break
            
            # Process imports and requires
            import_patterns = [patterns['import'], patterns['require']]
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    import_name = match.group(1)
                    import_node = self._create_node(
                        NodeType.VARIABLE, import_name, line.strip(),
                        self.current_file_path, i, i, 0, 0
                    )
                    self._create_edge(module_node, import_node, EdgeType.AST_CHILD)
                    break
            
            # Process variable declarations
            match = re.match(patterns['variable_decl'], line)
            if match:
                var_name = match.group(1)
                var_node = self._create_node(
                    NodeType.VARIABLE, var_name, line.strip(),
                    self.current_file_path, i, i, 0, 0
                )
                self._create_edge(module_node, var_node, EdgeType.AST_CHILD)
            
            # Process assignments
            match = re.match(patterns['assignment'], line)
            if match:
                assign_node = self._create_node(
                    NodeType.ASSIGNMENT, "assignment", line.strip(),
                    self.current_file_path, i, i, 0, 0
                )
                self._create_edge(module_node, assign_node, EdgeType.AST_CHILD)
                
                var_name = match.group(1)
                var_node = self._create_node(
                    NodeType.VARIABLE, var_name, var_name,
                    self.current_file_path, i, i, 0, 0
                )
                self._create_edge(assign_node, var_node, EdgeType.AST_CHILD)
            
            # Process control structures
            control_patterns = [
                (patterns['if_statement'], 'if'),
                (patterns['for_loop'], 'for'),
                (patterns['while_loop'], 'while'),
                (patterns['try_catch'], 'try'),
            ]
            
            for pattern, control_type in control_patterns:
                if re.match(pattern, line):
                    control_node = self._create_node(
                        NodeType.CONDITION, control_type, line.strip(),
                        self.current_file_path, i, i, 0, 0
                    )
                    self._create_edge(module_node, control_node, EdgeType.AST_CHILD)
                    break